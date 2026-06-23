"""GolfBot Main GUI — camera + 2D schematic with mode controls.

Tkinter GUI with two threads: a vision daemon thread captures frames and runs
the processing pipeline, while the main thread runs the tkinter event loop and
refreshes the display at ~30 fps independently of vision throughput.
"""

from __future__ import annotations

import json
import math
import threading
import time
import tkinter as tk
from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np
from PIL import Image, ImageTk

from brain.brain import BrainController
from brain.models import BrainState
from control.commander import RobotCommander
from control.telemetry import log_event
from control.spin_calibration import SpinController, SpinStatus
from guidance.guidance import GuidanceController, GuidanceStatus
from localization.localization import RobotCalibrationCollector, RobotPoseEstimator
from localization.models import RobotCalibrationRuntime, RobotMarkerObservation, RobotPose
from path.models import HybridPose, PlannedBallTarget
from path.planner import plan_route
from path.pickup_geometry import compute_pickup_geometry
from path.tools.pickup_visualizer import (
    OverlayMode,
    STRATEGY_OPTIONS,
    draw_pickup_geometry,
    draw_route_plan,
)
from config import AppConfig, RouteStrategyName
from perception.vision.debug import DebugRenderer
from perception.vision.cross_tracking import CrossAction, CrossCollisionTracker, relocalize_cross
from perception.vision.models import (
    CalibrationState,
    RedCrossSpec,
    RedZoneDetection,
    SmoothedBallCoordinate,
)
from perception.vision.pipeline import VisionPipeline, VisionFrameResult


class AppMode(str, Enum):
    IDLE = "IDLE"
    MANUAL = "MANUAL"
    AUTO = "AUTO"
    GUIDANCE_TEST = "GUIDANCE_TEST"
    CALIBRATE = "CALIBRATE"


# Robot self-calibration sub-phases (within AppMode.CALIBRATE)
CALIB_IDLE = "idle"
CALIB_CONNECTING = "connecting"
CALIB_SPIN = "spin"
CALIB_ALIGN = "align"

# Geometry nudge step sizes
GEOM_STEP_CM = 0.5
HEADING_STEP_RAD = math.radians(1.0)


WINDOW_NAME = "GolfBot Main"
CORNER_WINDOW_NAME = "Set Field Corners"
CROSS_WINDOW_NAME = "Place Red Cross"
ROUTE_VIEW_WINDOW_NAME = "Route View"

# ---------------------------------------------------------------------------
# Hardcoded test routes for Stage 2 guidance isolation testing
# ---------------------------------------------------------------------------

def _route_waypoints(coords: list[tuple[float, float]]) -> list[HybridPose]:
    """Build a waypoint list with theta pointing toward the next waypoint."""
    waypoints: list[HybridPose] = []
    for i, (x, y) in enumerate(coords):
        if i < len(coords) - 1:
            nx, ny = coords[i + 1]
            theta = math.atan2(ny - y, nx - x)
        else:
            theta = waypoints[-1].theta_rad if waypoints else 0.0
        waypoints.append(HybridPose(x_cm=x, y_cm=y, theta_rad=theta))
    return waypoints


TEST_ROUTES: dict[str, list[HybridPose]] = {
    "straight": _route_waypoints([(30, 60), (137, 60)]),
    "90_turn":  _route_waypoints([(40, 30), (120, 30), (120, 90)]),
    "L_shape":  _route_waypoints([(30, 30), (100, 30), (100, 90), (140, 90)]),
    "square": _route_waypoints([(30, 30), (100, 30), (100, 90), (30, 90), (30, 30)]),
    "s": _route_waypoints([(30, 30), (137, 30), (137, 61), (30, 61), (30, 92), (137, 92)]),
    "lightning": _route_waypoints([(30, 30), (65, 92), (100, 30), (137, 92), (137, 30)]),
    "wave": _route_waypoints([(30, 61), (55, 30), (80, 92), (105, 30), (137, 61)]),
    "star": _route_waypoints([(84, 30), (100, 68), (137, 68), (107, 92), (119, 61), (84, 80), (49, 61), (61, 92), (31, 68), (68, 68), (84, 30)]),
}

TEST_ROUTE_NAMES: list[str] = list(TEST_ROUTES.keys())

ROUTE_VIEW_KEY_BY_CONFIG_STRATEGY: dict[RouteStrategyName, str] = {
    RouteStrategyName.SET_COVER_NEAREST: "set-cover",
    RouteStrategyName.INTERSECTION_PRIORITY: "intersections",
    RouteStrategyName.INTERSECTION_NEAREST: "intersection-nearest",
    RouteStrategyName.INTERSECTION_OPTIMAL: "intersection-optimal",
}

CONFIG_STRATEGY_BY_ROUTE_VIEW_KEY: dict[str, RouteStrategyName] = {
    value: key for key, value in ROUTE_VIEW_KEY_BY_CONFIG_STRATEGY.items()
}


# ---------------------------------------------------------------------------
# Corner selection — temporary window for picking 4 field corners
# ---------------------------------------------------------------------------

@dataclass
class CornerSelectionState:
    """State for the temporary corner-selection window."""
    points: list[tuple[int, int]] = field(default_factory=list)
    cursor: tuple[int, int] = (0, 0)
    frame_size: tuple[int, int] = (0, 0)
    done: bool = False
    cancelled: bool = False

    def clear(self) -> None:
        self.points.clear()
        self.done = False
        self.cancelled = False


def _draw_loupe(overlay: np.ndarray, cursor: tuple[int, int],
                crop_sz: int = 40, scale: int = 5, padding: int = 12) -> None:
    """Draw a magnified crosshair loupe of the area under the cursor (top-right)."""
    h, w = overlay.shape[:2]
    crop_w = min(crop_sz, w)
    crop_h = min(crop_sz, h)
    cx, cy = cursor
    x0 = max(0, cx - crop_w // 2)
    x1 = x0 + crop_w
    if x1 > w:
        x1 = w
        x0 = x1 - crop_w
    y0 = max(0, cy - crop_h // 2)
    y1 = y0 + crop_h
    if y1 > h:
        y1 = h
        y0 = y1 - crop_h
    crop = overlay[y0:y1, x0:x1]
    loupe = cv2.resize(crop, (crop.shape[1] * scale, crop.shape[0] * scale),
                       interpolation=cv2.INTER_NEAREST)
    lh, lw = loupe.shape[:2]
    dx1 = w - padding
    dx0 = max(0, dx1 - lw)
    dy0 = padding
    dy1 = min(h, dy0 + lh)
    vis = loupe[:dy1 - dy0, :dx1 - dx0]
    overlay[dy0:dy1, dx0:dx1] = vis
    cv2.rectangle(overlay, (dx0, dy0), (dx1, dy1), (255, 255, 255), 2)
    lcx = dx0 + vis.shape[1] // 2
    lcy = dy0 + vis.shape[0] // 2
    cv2.line(overlay, (lcx, dy0), (lcx, dy1), (0, 255, 255), 1)
    cv2.line(overlay, (dx0, lcy), (dx1, lcy), (0, 255, 255), 1)


def _draw_corner_overlay(frame: np.ndarray, state: CornerSelectionState) -> np.ndarray:
    """Draw point markers, polylines, loupe, and help text on the selector view."""
    overlay = frame.copy()
    h, w = overlay.shape[:2]

    _draw_loupe(overlay, state.cursor)

    # Points
    for i, pt in enumerate(state.points, start=1):
        cv2.circle(overlay, pt, 6, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(overlay, pt, 10, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(overlay, str(i), (pt[0] + 10, pt[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    if len(state.points) >= 2:
        poly = np.array(state.points, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(overlay, [poly], False, (0, 255, 0), 2, cv2.LINE_AA)
    if len(state.points) == 4:
        corners = np.array(state.points, dtype=np.float32)
        sums = corners.sum(axis=1)
        diffs = np.diff(corners, axis=1).reshape(4)
        ordered = np.array([
            corners[np.argmin(sums)], corners[np.argmin(diffs)],
            corners[np.argmax(sums)], corners[np.argmax(diffs)],
        ], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(overlay, [ordered], True, (255, 200, 0), 2, cv2.LINE_AA)

    # Help text
    n = len(state.points)
    lines = [
        f"Points: {n}/4",
        "Left click: add corner",
        "Right click / r: reset",
        "q / Esc: cancel",
    ]
    for i, text in enumerate(lines):
        cv2.putText(overlay, text, (16, 30 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    status = "Click the 4 inner field corners" if n < 4 else "Corners set — closing"
    color = (0, 165, 255) if n < 4 else (0, 255, 0)
    cv2.putText(overlay, status, (16, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
    return overlay


# ---------------------------------------------------------------------------
# Shared display state for vision→GUI thread handoff
# ---------------------------------------------------------------------------

@dataclass
class SharedDisplayState:
    """Data published by the vision thread, consumed by the GUI thread."""
    left_panel: np.ndarray | None = None
    right_panel: np.ndarray | None = None
    corner_image: np.ndarray | None = None
    cross_image: np.ndarray | None = None
    route_image: np.ndarray | None = None
    frame_seq: int = 0
    fps: float = 0.0
    mode: str = "IDLE"
    message: str = "Ready"
    robot_pose: RobotPose | None = None
    brain_state: str | None = None
    guidance_status: str | None = None
    calibration_phase: str = CALIB_IDLE
    # Extra status fields for the status bar
    cal_state: str = ""
    connecting: bool = False
    commander_connected: bool = False
    active_route_name: str = ""
    guidance_cursor: int = 0
    guidance_wp_count: int = 0
    brain_step_cursor: int = 0
    brain_step_count: int = 0
    timer_elapsed: float = 0.0
    markers_visible: str = ""
    corner_window_open: bool = False
    cross_window_open: bool = False
    route_view_open: bool = False


# ---------------------------------------------------------------------------
# Main GUI
# ---------------------------------------------------------------------------

@dataclass
class MainGui:
    """Two-thread GUI: vision processing in daemon thread, display in main thread."""

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
    _corner_state: CornerSelectionState = field(default_factory=CornerSelectionState)
    _corner_window_open: bool = False

    # Guidance test state
    _commander: RobotCommander | None = None
    _guidance: GuidanceController | None = None
    _connecting: bool = False
    _guidance_status: GuidanceStatus | None = None
    _active_route_name: str = "L_shape"
    _active_route: list[HybridPose] | None = None
    _last_guidance_time: float | None = None

    # Brain / Auto state
    _brain: BrainController | None = None
    _brain_state: BrainState | None = None
    _last_brain_time: float | None = None
    _brain_route_points: list[HybridPose] | None = None
    _last_result: VisionFrameResult | None = None

    # Crop monitor state (ROI-based ball tracking during AUTO mode)
    _tracked_balls: list | None = None        # fixed positions from last full scan
    _crop_missing_counts: dict = field(default_factory=dict)  # track_id → consecutive missing frames
    _last_crop_missing: set = field(default_factory=set)      # track_ids missing on last check
    _crop_hsv_ranges: dict = field(default_factory=dict)      # track_id → (lower, upper) per-ball HSV

    # Pickup verification state (deferred YOLO snapshot after cluster exit)
    _pending_verification: list = field(default_factory=list)  # balls awaiting post-pickup YOLO check
    _attempted_pickups: int = 0                                 # count of attempts in current pending batch
    _needs_verification_snapshot: bool = False                  # triggers YOLO on next _process_frame
    _verification_snapshot_done: bool = False                   # set by _process_frame, consumed by verifier

    # Timer state
    _timer_start_time: float | None = None
    _timer_elapsed: float = 0.0
    _timer_running: bool = False
    _frames_elapsed: int = 0

    # Red-cross collision re-localization (fire-on-exit state machine)
    _cross_tracker: CrossCollisionTracker = field(default_factory=CrossCollisionTracker)

    # Robot self-calibration state
    _calib_collector: RobotCalibrationCollector | None = None
    _calib_runtime: RobotCalibrationRuntime | None = None
    _spin: SpinController | None = None
    _calib_phase: str = CALIB_IDLE
    _calib_backup: dict | None = None
    _latest_observations: dict[int, RobotMarkerObservation] = field(default_factory=dict)
    _latest_parallax: object | None = None

    # Manual red-cross obstacle state
    _cross_spec: RedCrossSpec | None = None
    _cross_window_open: bool = False
    _cross_state: CornerSelectionState = field(default_factory=CornerSelectionState)

    # Overlay mode for right-panel heatmap/collision visualization
    _overlay_mode: OverlayMode = OverlayMode.NONE

    # Route View window state
    _route_view_open: bool = False
    _route_view_strategy_index: int = 0
    _route_view_cache_id: int = 0
    _route_view_cached_image: np.ndarray | None = None

    # Dimensions derived from config
    _left_w: int = 0
    _left_h: int = 0
    _right_w: int = 0
    _right_h: int = 0

    # Threading state (initialized in __post_init__)
    _display_lock: threading.Lock = field(default_factory=threading.Lock)
    _params_lock: threading.Lock = field(default_factory=threading.Lock)
    _shared: SharedDisplayState = field(default_factory=SharedDisplayState)
    _frame_seq: int = 0

    def __post_init__(self) -> None:
        self._left_w, self._left_h = self.config.camera.topdown_warp_size
        self._right_w = self.config.windows.schematic_width_px
        self._right_h = self.config.windows.schematic_height_px
        self.params = self.pipeline.default_params()
        self._seed_geometry_params()
        self._load_robot_calibration()
        self._load_field_corners()
        self._load_cross()
        self._route_view_strategy_index = self._route_view_strategy_index_from_config()
        # Route planning uses the standalone plan_route() facade from path.pathfinding.

    def _load_field_corners(self) -> None:
        """Restore the saved manual top-down warp, if a corners file exists."""
        corners_path = self.config.paths.field_corners_file
        if corners_path is None or not corners_path.exists():
            return
        calibrator = self.pipeline.preprocessor.homography_calibrator
        if calibrator.load_manual_corners(corners_path):
            self.message = f"Field corners loaded from {corners_path.name} — warp active"
        else:
            self.message = "Saved field corners file invalid — press Set Corners"

    # ------------------------------------------------------------------
    # Manual red-cross placement (click tip corner, then armpit corner)
    # ------------------------------------------------------------------

    def _start_cross_placement(self) -> None:
        """Open the zoomed cross-placement window (top-down view + loupe)."""
        if self.pipeline.preprocessor.homography_calibrator.transform_matrix is None:
            self.message = "Set field corners before placing the cross"
            return
        if self.camera is None and self.static_image is None:
            self.message = "No camera — cannot place cross"
            return
        self._cross_state.clear()
        self._cross_window_open = True

        top = tk.Toplevel(self._root)
        top.title(CROSS_WINDOW_NAME)
        top.resizable(False, False)
        top.protocol("WM_DELETE_WINDOW", lambda: self._close_cross_window("Cross placement cancelled"))
        self._cross_toplevel = top

        canvas = tk.Canvas(top, bg="#000000")
        canvas.pack()
        self._cross_canvas = canvas
        self._cross_photo: ImageTk.PhotoImage | None = None

        def on_click(event):
            state = self._cross_state
            w, h = state.frame_size
            if w > 0 and h > 0:
                state.cursor = (int(np.clip(event.x, 0, w - 1)), int(np.clip(event.y, 0, h - 1)))
            if len(state.points) < 2:
                state.points.append(state.cursor)
                if len(state.points) == 2:
                    state.done = True

        def on_right_click(event):
            self._cross_state.points.clear()
            self._cross_state.done = False

        def on_motion(event):
            state = self._cross_state
            w, h = state.frame_size
            if w > 0 and h > 0:
                state.cursor = (int(np.clip(event.x, 0, w - 1)), int(np.clip(event.y, 0, h - 1)))

        canvas.bind("<Button-1>", on_click)
        canvas.bind("<Button-2>", on_right_click)  # macOS right-click
        canvas.bind("<Button-3>", on_right_click)
        canvas.bind("<Motion>", on_motion)
        self.message = "Cross window: click an arm TIP corner, then its inner ARMPIT corner"

    def _close_cross_window(self, msg: str | None = None) -> None:
        self._cross_window_open = False
        if msg:
            self.message = msg
        if hasattr(self, "_cross_toplevel") and self._cross_toplevel is not None:
            try:
                self._cross_toplevel.destroy()
            except tk.TclError:
                pass
            self._cross_toplevel = None

    def _render_cross_image(self, raw_frame: np.ndarray) -> np.ndarray | None:
        """Render the cross-placement overlay and return the image, or None if warp unavailable."""
        # Show the top-down (warped) view, where the cross is rectified and pixels
        # map linearly to field cm.
        undistorted = self.pipeline.preprocessor.undistort(raw_frame)
        topdown = self.pipeline.preprocessor.homography_calibrator.warp(undistorted)
        if topdown is None:
            self.message = "No top-down warp — set field corners first"
            self._close_cross_window()
            return None

        state = self._cross_state
        state.frame_size = (topdown.shape[1], topdown.shape[0])
        if state.cursor == (0, 0):
            state.cursor = (topdown.shape[1] // 2, topdown.shape[0] // 2)

        view = topdown.copy()
        _draw_loupe(view, state.cursor)

        # Live preview: anchor + (second click or current cursor)
        if len(state.points) >= 1:
            tip_px = state.points[0]
            armpit_px = state.points[1] if len(state.points) == 2 else state.cursor
            self._draw_cross_preview(view, tip_px, armpit_px)

        labels = ["TIP corner", "ARMPIT corner"]
        for i, point in enumerate(state.points):
            cv2.circle(view, point, 5, (255, 0, 0), -1, cv2.LINE_AA)
            cv2.putText(view, labels[i] if i < len(labels) else "", (point[0] + 8, point[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2, cv2.LINE_AA)

        help_lines = [
            f"Points: {len(state.points)}/2",
            "1) Click an arm TIP corner",
            "2) Click that arm's inner ARMPIT corner",
            "Size scales with the gap. Right click/r: reset. q/Esc: cancel",
        ]
        for i, text in enumerate(help_lines):
            cv2.putText(view, text, (16, 28 + i * 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        if state.done:
            tip_cm = self.pipeline.mapper.topdown_px_to_field_cm(state.points[0])
            armpit_cm = self.pipeline.mapper.topdown_px_to_field_cm(state.points[1])
            self._cross_spec = RedCrossSpec.from_tip_and_armpit(tip_cm, armpit_cm)
            self._save_cross()
            self._close_cross_window()

        return view

    def _draw_cross_preview(self, image: np.ndarray, tip_px: tuple[int, int], armpit_px: tuple[int, int]) -> None:
        """Draw a cross preview in the window from two top-down pixel corners."""
        mapper = self.pipeline.mapper
        spec = RedCrossSpec.from_tip_and_armpit(
            mapper.topdown_px_to_field_cm(tip_px),
            mapper.topdown_px_to_field_cm(armpit_px),
        )
        poly = np.array(
            [mapper.field_cm_to_topdown_pixel(p) for p in spec.polygon_cm()],
            dtype=np.int32,
        ).reshape(-1, 1, 2)
        cv2.polylines(image, [poly], True, (0, 0, 255), 2, cv2.LINE_AA)
        center = mapper.field_cm_to_topdown_pixel(spec.center_cm)
        cv2.drawMarker(image, (int(round(center[0])), int(round(center[1]))),
                       (0, 255, 255), cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)

    def _save_cross(self) -> None:
        path = self.config.paths.red_cross_file
        if self._cross_spec is None or path is None:
            return
        try:
            path.write_text(json.dumps(self._cross_spec.to_dict(), indent=2), encoding="utf-8")
            cx, cy = self._cross_spec.center_cm
            self.message = f"Red cross placed at ({cx:.1f}, {cy:.1f}) cm — saved to {path.name}"
        except OSError as exc:
            self.message = f"Red cross placed (save failed: {exc})"

    def _load_cross(self) -> None:
        """Restore the saved red cross, if a file exists."""
        path = self.config.paths.red_cross_file
        if path is None or not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        spec = RedCrossSpec.from_dict(data)
        if spec is not None:
            self._cross_spec = spec
            self.message = f"Red cross loaded from {path.name}"

    def cross_red_zone(self) -> RedZoneDetection | None:
        """Expose the placed cross as a RedZoneDetection for occupancy/logic.

        Integration hook: appending this to ``VisionFrameResult.red_zones`` before
        the occupancy grid is built makes the manual cross an avoidance obstacle.
        """
        if self._cross_spec is None:
            return None
        mapper = self.pipeline.mapper
        points = np.array(
            [mapper.field_cm_to_topdown_pixel(p) for p in self._cross_spec.polygon_cm()],
            dtype=np.int32,
        )
        contour = points.reshape(-1, 1, 2)
        x, y, w, h = cv2.boundingRect(points)
        cx, cy = mapper.field_cm_to_topdown_pixel(self._cross_spec.center_cm)
        center_px = (int(round(cx)), int(round(cy)))
        return RedZoneDetection(
            contour=contour,
            corrected_contour=contour,
            bounding_box=(int(x), int(y), int(w), int(h)),
            center=center_px,
            corrected_center=center_px,
            area=float(cv2.contourArea(points)),
        )

    def _seed_geometry_params(self) -> None:
        """Ensure live geometry keys exist in params, defaulting from config."""
        robot = self.config.robot
        planner = self.config.planner
        self.params.setdefault("robot_width_cm", robot.tuned_footprint_width_cm)
        self.params.setdefault("robot_front_cm", robot.tuned_footprint_front_from_origin_cm)
        self.params.setdefault("robot_rear_cm", robot.tuned_footprint_rear_from_origin_cm)
        self.params.setdefault("tube_forward_cm", robot.tuned_tube_offset_cm)
        self.params.setdefault("tube_right_cm", robot.tuned_tube_right_offset_cm)
        self.params.setdefault("tube_width_cm", planner.tube_width_cm)
        self.params.setdefault("mouth_radius_cm", planner.mouth_radius_cm)
        self.params.setdefault("unload_extension_cm", planner.unload_extension_cm)
        self.params.setdefault("pipe_diameter_cm", planner.pipe_diameter_cm)
        self.params.setdefault("heading_tuning_rad", 0.0)
        # Crop monitor HSV params (white ball detection in fixed crops)
        self.params.setdefault("crop_size", 60)
        self.params.setdefault("crop_white_h_min", 0)
        self.params.setdefault("crop_white_h_max", 180)
        self.params.setdefault("crop_white_s_min", 0)
        self.params.setdefault("crop_white_s_max", 40)
        self.params.setdefault("crop_white_v_min", 200)
        self.params.setdefault("crop_white_v_max", 255)
        self.params.setdefault("crop_min_pixel_fraction", 0.03)
        self.params.setdefault("crop_missing_threshold", 5)

    def _load_robot_calibration(self) -> None:
        cal_path = self.config.paths.robot_calibration_file
        if cal_path is not None and cal_path.exists():
            collector = RobotCalibrationCollector(self.config.robot)
            self.calibration = collector.load_robot_calibration(
                cal_path, self.config.camera.topdown_warp_size,
            )
            if self.calibration is not None:
                RobotCalibrationCollector.apply_geometry_to_params(
                    self.params, self.calibration.get("geometry"),
                )
                self.message = "Robot calibration loaded"
            else:
                self.message = "Robot calibration file found but invalid"
        else:
            self.message = "No robot calibration file"

    _GEOMETRY_NUDGES = {
        "front_inc": ("robot_front_cm", GEOM_STEP_CM),
        "front_dec": ("robot_front_cm", -GEOM_STEP_CM),
        "rear_inc": ("robot_rear_cm", GEOM_STEP_CM),
        "rear_dec": ("robot_rear_cm", -GEOM_STEP_CM),
        "width_inc": ("robot_width_cm", GEOM_STEP_CM),
        "width_dec": ("robot_width_cm", -GEOM_STEP_CM),
        "pipe_inc": ("tube_forward_cm", GEOM_STEP_CM),
        "pipe_dec": ("tube_forward_cm", -GEOM_STEP_CM),
        "head_inc": ("heading_tuning_rad", HEADING_STEP_RAD),
        "head_dec": ("heading_tuning_rad", -HEADING_STEP_RAD),
    }

    def _handle_button(self, action: str) -> None:
        if action in self._GEOMETRY_NUDGES:
            param_key, delta = self._GEOMETRY_NUDGES[action]
            self._geometry_nudge(param_key, delta)
            return
        if action == "set_corners":
            self._open_corner_window()
        elif action == "set_cross":
            self._start_cross_placement()
        elif action == "calib_robot":
            self._start_calibration()
        elif action == "calib_spin":
            self._start_spin()
        elif action == "calib_save":
            self._save_calibration()
        elif action == "calib_cancel":
            self._cancel_calibration()
        elif action == "route_view":
            self._open_route_view()
        elif action == "guidance_test":
            if self._block_during_calibration():
                return
            if self.mode == AppMode.GUIDANCE_TEST:
                self._cycle_test_route()
            else:
                self._start_guidance_test()
        elif action == "manual":
            if self._block_during_calibration():
                return
            self._disconnect_guidance()
            self.mode = AppMode.MANUAL
            self.message = "Manual mode (view only)"
        elif action == "auto":
            if self._block_during_calibration():
                return
            self._start_brain()
        elif action == "stop":
            if self.mode == AppMode.CALIBRATE:
                self._cancel_calibration()
            else:
                self._disconnect_guidance()
                self.mode = AppMode.IDLE
                self.message = "Stopped"
        elif action == "quit":
            self.closed = True

    def _block_during_calibration(self) -> bool:
        """Refuse mode switches while a calibration is in progress."""
        if self.mode == AppMode.CALIBRATE:
            self.message = "Finish or cancel robot calibration first"
            return True
        return False

    # ------------------------------------------------------------------
    # Corner selection window
    # ------------------------------------------------------------------

    def _open_corner_window(self) -> None:
        if self.camera is None and self.static_image is None:
            self.message = "No camera — cannot set corners"
            return
        self._corner_state.clear()
        self._corner_window_open = True

        top = tk.Toplevel(self._root)
        top.title(CORNER_WINDOW_NAME)
        top.resizable(False, False)
        top.protocol("WM_DELETE_WINDOW", lambda: self._close_corner_window("Corner selection cancelled"))
        self._corner_toplevel = top

        canvas = tk.Canvas(top, bg="#000000")
        canvas.pack()
        self._corner_canvas = canvas
        self._corner_photo: ImageTk.PhotoImage | None = None

        def on_click(event):
            state = self._corner_state
            w, h = state.frame_size
            if w > 0 and h > 0:
                state.cursor = (int(np.clip(event.x, 0, w - 1)), int(np.clip(event.y, 0, h - 1)))
            if len(state.points) < 4:
                state.points.append(state.cursor)
                if len(state.points) == 4:
                    state.done = True

        def on_right_click(event):
            self._corner_state.points.clear()

        def on_motion(event):
            state = self._corner_state
            w, h = state.frame_size
            if w > 0 and h > 0:
                state.cursor = (int(np.clip(event.x, 0, w - 1)), int(np.clip(event.y, 0, h - 1)))

        canvas.bind("<Button-1>", on_click)
        canvas.bind("<Button-2>", on_right_click)  # macOS right-click
        canvas.bind("<Button-3>", on_right_click)
        canvas.bind("<Motion>", on_motion)
        self.message = "Corner selection window opened — click 4 inner field corners"

    def _close_corner_window(self, msg: str | None = None) -> None:
        self._corner_window_open = False
        if msg:
            self.message = msg
        if hasattr(self, "_corner_toplevel") and self._corner_toplevel is not None:
            try:
                self._corner_toplevel.destroy()
            except tk.TclError:
                pass
            self._corner_toplevel = None

    def _render_corner_image(self, raw_frame: np.ndarray) -> np.ndarray | None:
        """Render the corner-selection overlay and return the image, or None if not ready."""
        # Undistort (but do NOT warp) so the user sees the real camera with
        # barrel distortion removed — the same view the homography is built on.
        undistorted = self.pipeline.preprocessor.undistort(raw_frame)
        self._corner_state.frame_size = (undistorted.shape[1], undistorted.shape[0])
        if self._corner_state.cursor == (0, 0):
            self._corner_state.cursor = (undistorted.shape[1] // 2, undistorted.shape[0] // 2)

        view = _draw_corner_overlay(undistorted, self._corner_state)

        if self._corner_state.done:
            # Feed the 4 corners into the pipeline's HomographyCalibrator and persist
            # them so the warp is restored automatically on the next launch.
            calibrator = self.pipeline.preprocessor.homography_calibrator
            calibrator.set_manual_points(self._corner_state.points)
            corners_path = self.config.paths.field_corners_file
            try:
                calibrator.save_manual_corners(corners_path, self._corner_state.frame_size)
                self.message = f"Field corners set and saved to {corners_path.name} — warp active"
            except OSError as exc:
                self.message = f"Field corners set — warp active (save failed: {exc})"
            self._close_corner_window()

        return view

    # ------------------------------------------------------------------
    # Guidance test (Stage 2 isolation testing)
    # ------------------------------------------------------------------

    def _start_guidance_test(self) -> None:
        """Connect to the robot in a background thread and load a test route."""
        if self._connecting:
            self.message = "Already connecting..."
            return
        if self._commander is not None:
            # Already connected — just load route
            self._load_test_route(self._active_route_name)
            return
        self._connecting = True
        self.message = "Connecting to robot..."

        def connect() -> None:
            try:
                commander = RobotCommander(
                    connection_config=self.config.connection,
                    drive_config=self.config.drive,
                    auto_connect=True,
                )
                self._commander = commander
                self._guidance = GuidanceController(commander, config=self.config.drive)
                self._load_test_route(self._active_route_name)
                self.message = f"Connected — route: {self._active_route_name}"
            except Exception as exc:
                self.message = f"Connection failed: {exc}"
                self._commander = None
                self._guidance = None
            finally:
                self._connecting = False

        thread = threading.Thread(target=connect, daemon=True)
        thread.start()

    def _load_test_route(self, name: str) -> None:
        """Set a test route on the guidance controller."""
        self._active_route_name = name
        self._active_route = TEST_ROUTES[name]
        self._last_guidance_time = None
        self._guidance_status = None
        if self._guidance is not None:
            self._guidance.set_route(list(self._active_route))
        self.mode = AppMode.GUIDANCE_TEST
        self.message = f"Guidance test — route: {name}"

    def _cycle_test_route(self) -> None:
        """Advance to the next test route."""
        idx = TEST_ROUTE_NAMES.index(self._active_route_name)
        next_name = TEST_ROUTE_NAMES[(idx + 1) % len(TEST_ROUTE_NAMES)]
        self._load_test_route(next_name)

    def _disconnect_guidance(self) -> None:
        """Clear route, stop robot, close socket, reset guidance and brain state."""
        self._timer_running = False

        if self._brain is not None:
            self._brain.reset()
        self._brain = None
        self._brain_state = None
        self._brain_route_points = None
        self._last_brain_time = None
        self._tracked_balls = None
        self._crop_missing_counts = {}
        self._last_crop_missing = set()
        self._crop_hsv_ranges = {}
        self._pending_verification = []
        self._attempted_pickups = 0
        self._needs_verification_snapshot = False
        self._verification_snapshot_done = False
        self._cross_tracker.reset()
        if self._guidance is not None:
            self._guidance.clear_route()
        if self._commander is not None:
            try:
                self._commander.close()
            except Exception:
                pass
        self._commander = None
        self._guidance = None
        self._guidance_status = None
        self._active_route = None
        self._last_guidance_time = None

    def _tick_guidance(self) -> None:
        """Run one guidance frame if in GUIDANCE_TEST mode."""
        if self.mode != AppMode.GUIDANCE_TEST:
            return
        if self._guidance is None or self._connecting:
            return

        now = time.perf_counter()
        if self._last_guidance_time is None:
            dt_s = 0.033
        else:
            dt_s = max(0.001, min(0.5, now - self._last_guidance_time))
        self._last_guidance_time = now

        self._guidance_status = self._guidance.tick(self.robot_pose, dt_s)

    # ------------------------------------------------------------------
    # Brain / Auto mode
    # ------------------------------------------------------------------

    def _start_brain(self) -> None:
        """Validate inputs, plan a route, and start the Brain FSM."""
        if self._connecting:
            self.message = "Already connecting..."
            return
        if self._brain is not None:
            self.message = "Brain already running"
            return

        # Validate prerequisites
        if self.robot_pose is None:
            self.message = "Cannot start: no robot pose"
            return
        result = self._last_result
        if result is None or result.occupancy_grid is None:
            self.message = "Cannot start: no occupancy grid"
            return
        if not result.smoothed_ball_coordinates:
            self.message = "Cannot start: no balls detected"
            return

        # Capture current frame data before spawning thread
        captured_grid = result.occupancy_grid.copy()
        captured_balls = list(result.smoothed_ball_coordinates)
        captured_pose = self.robot_pose
        captured_frame = result.frame_for_detection

        self._connecting = True
        self.message = "Connecting and planning route..."

        def connect_and_plan() -> None:
            try:
                commander = RobotCommander(
                    connection_config=self.config.connection,
                    drive_config=self.config.drive,
                    auto_connect=True,
                )
                guidance = GuidanceController(commander, config=self.config.drive)
                brain = BrainController(guidance, commander)

                start_pose = HybridPose(
                    x_cm=captured_pose.x_cm,
                    y_cm=captured_pose.y_cm,
                    theta_rad=captured_pose.heading_rad,
                )
                targets = [
                    PlannedBallTarget(
                        track_id=b.track_id,
                        label=b.label,
                        x_cm=b.cm_x,
                        y_cm=b.cm_y,
                        node_cm=self.pipeline.mapper.field_metric_cm_to_grid_node(
                            (b.cm_x, b.cm_y),
                        ),
                    )
                    for b in captured_balls
                ]
                geometry = self.pose_estimator.robot_geometry_from_params(self.params)

                unload_reach = geometry.rear_cm + geometry.unload_extension_cm
                unload_pos = (unload_reach + 2.0, self.config.field.height_cm * 0.5)
                plan = plan_route(
                    captured_grid,
                    targets,
                    start_pose,
                    geometry,
                    self.config.field,
                    unload_position=unload_pos,
                    obstacle_margin_cm=self.config.field.obstacle_margin_cm,
                )

                if not plan.waypoints:
                    self.message = "Planner returned empty route"
                    commander.close()
                    return

                brain.load_route(plan)

                self._commander = commander
                self._guidance = guidance
                self._brain = brain
                self._brain_route_points = [
                    HybridPose(w.x_cm, w.y_cm, w.theta_rad) for w in plan.waypoints
                ]
                self._last_brain_time = None
                self._brain_state = None
                self._tracked_balls = list(captured_balls)
                self._crop_missing_counts = {}
                self._crop_hsv_ranges = (
                    self.pipeline.calibrate_crop_hsv(captured_frame, captured_balls, self.params)
                    if captured_frame is not None else {}
                )
                self.mode = AppMode.AUTO
                self.message = f"Brain running — {brain.step_count} steps"

                self._timer_start_time = time.perf_counter()
                self._timer_elapsed = 0.0
                self._timer_running = True
            except Exception as exc:
                self.message = f"Brain start failed: {exc}"
                self._commander = None
                self._guidance = None
                self._brain = None
            finally:
                self._connecting = False

        threading.Thread(target=connect_and_plan, daemon=True).start()

    def _tick_brain(self) -> None:
        """Run one Brain FSM frame if in AUTO mode."""
        if self.mode != AppMode.AUTO:
            return
        if self._brain is None or self._connecting:
            return

        now = time.perf_counter()
        if self._last_brain_time is None:
            dt_s = 0.033
        else:
            dt_s = max(0.001, min(0.5, now - self._last_brain_time))
        self._last_brain_time = now

        if self._timer_running and self._timer_start_time is not None:
            self._timer_elapsed = now - self._timer_start_time

            if self._brain_state and self._brain_state.name == "DONE":
                self._timer_running = False

        prev_state = self._brain_state
        self._brain_state = self._brain.tick(self.robot_pose, dt_s)

        if prev_state == BrainState.PICKUP and self._brain_state == BrainState.IDLE:
            self._remove_collected_ball()

        
        if (self._frames_elapsed % 60 == 0): #Hvis ikke alle bolde er fundet til at starte med, så får de ikke et hsv crop.
            self._needs_verification_snapshot = True
            #Check for any missed pickups every 60 frames (2 seconds at 30fps) to catch any balls that were displaced before the crop monitor could detect them. Also gives a chance to catch any missed pickups after the brain is done, before stopping
        self._frames_elapsed += 1


        if (
            self._brain_state == BrainState.ERROR
            and self._brain.error_message == "ball_displaced"
            and self._last_result is not None
            and self._last_result.smoothed_ball_coordinates
            and self._last_result.occupancy_grid is not None
            and self.robot_pose is not None
        ):
            self._replan_after_displacement()

        if (self._brain_state == BrainState.DONE
            and self._last_result.smoothed_ball_coordinates
            and self._last_result.occupancy_grid is not None
            and self.robot_pose is not None
        ): #If brain is done, verify pickups to check for any missed balls before stopping
            self._needs_verification_snapshot = True
            #Could add a return 0 here to end program or someshit. Maybe move dance to here?



    def _remove_collected_ball(self) -> None:
        """Stage the nearest tracked ball for deferred YOLO pickup verification."""
        if not self._tracked_balls or self.robot_pose is None:
            return
        nearest = min(
            self._tracked_balls,
            key=lambda b: (b.cm_x - self.robot_pose.x_cm) ** 2 + (b.cm_y - self.robot_pose.y_cm) ** 2,
        )
        self._tracked_balls = [b for b in self._tracked_balls if b.track_id != nearest.track_id]
        self._crop_missing_counts.pop(nearest.track_id, None)
        self._crop_hsv_ranges.pop(nearest.track_id, None)
        self._pending_verification.append(nearest)
        self._attempted_pickups += 1
        log_event("BRAIN", "pickup staged for verification", track_id=nearest.track_id)

    def _tick_crop_monitor(self) -> None:
        """Check fixed HSV crops for each tracked ball; trigger rescan if any are missing."""
        if self.mode != AppMode.AUTO or self._tracked_balls is None:
            return
        result = self._last_result
        if result is None or result.frame_for_detection is None:
            return

        robot_xy = (
            (self.robot_pose.x_cm, self.robot_pose.y_cm) if self.robot_pose is not None else None
        )
        robot_radius = float(self.params.get("robot_radius_cm", 30.0))
        threshold = int(self.params.get("crop_missing_threshold", 5))

        missing_now = self.pipeline.check_ball_crops(
            result.frame_for_detection,
            self._tracked_balls,
            self.params,
            robot_pose_cm=robot_xy,
            robot_radius_cm=robot_radius,
            per_ball_hsv=self._crop_hsv_ranges,
        )
        self._last_crop_missing = missing_now

        for ball in self._tracked_balls:
            tid = ball.track_id
            if tid in missing_now:
                self._crop_missing_counts[tid] = self._crop_missing_counts.get(tid, 0) + 1
            else:
                self._crop_missing_counts[tid] = 0

        if any(c >= threshold for c in self._crop_missing_counts.values()):
            self._tracked_balls = None
            self._crop_missing_counts = {}
            if self._brain is not None:
                self._brain.signal_ball_displaced()

    def _ball_targets(self, balls: list[SmoothedBallCoordinate]) -> list[PlannedBallTarget]:
        """Build planner ball targets from smoothed field coordinates."""
        return [
            PlannedBallTarget(
                track_id=b.track_id,
                label=b.label,
                x_cm=b.cm_x,
                y_cm=b.cm_y,
                node_cm=self.pipeline.mapper.field_metric_cm_to_grid_node((b.cm_x, b.cm_y)),
            )
            for b in balls
        ]

    def _plan_and_load_route(self, targets: list[PlannedBallTarget]) -> bool:
        """Plan from the current pose through ``targets`` and load it into the brain.

        Returns True if a route was found and loaded.  Callers own their own
        status messages and any post-replan bookkeeping.
        """
        result = self._last_result
        if (
            result is None
            or result.occupancy_grid is None
            or self.robot_pose is None
            or self._brain is None
        ):
            return False
        start_pose = HybridPose(
            x_cm=self.robot_pose.x_cm,
            y_cm=self.robot_pose.y_cm,
            theta_rad=self.robot_pose.heading_rad,
        )
        geometry = self.pose_estimator.robot_geometry_from_params(self.params)
        unload_reach = geometry.rear_cm + geometry.unload_extension_cm
        unload_pos = (unload_reach + 2.0, self.config.field.height_cm * 0.5)
        plan = plan_route(
            result.occupancy_grid,
            targets,
            start_pose,
            geometry,
            self.config.field,
            unload_position=unload_pos,
            obstacle_margin_cm=self.config.field.obstacle_margin_cm,
        )

        if not plan.waypoints:
            self.message = "Rescan: no route found — stopping"
            return False

        self._brain.load_route(plan)
        self._brain_route_points = [
            HybridPose(w.x_cm, w.y_cm, w.theta_rad) for w in plan.waypoints
        ]

        return True

    def _replan_after_displacement(self) -> None:
        """Replan route from the current snapshot after a ball-displaced error."""
        result = self._last_result
        if not self._plan_and_load_route(self._ball_targets(result.smoothed_ball_coordinates)):
            self.message = "Rescan: no route found — stopping"
            return

        self._tracked_balls = list(result.smoothed_ball_coordinates)
        self._crop_missing_counts = {}
        self._crop_hsv_ranges = (
            self.pipeline.calibrate_crop_hsv(result.frame_for_detection, result.smoothed_ball_coordinates, self.params)
            if result.frame_for_detection is not None else {}
        )
        self.message = f"Replanned — {self._brain.step_count} steps"
        log_event("BRAIN", "replanned after rescan", steps=self._brain.step_count)

    def _tick_pickup_verifier(self) -> None:
        """Trigger a YOLO snapshot when the robot has moved away from all pending pickups."""
        if self.mode != AppMode.AUTO or self._brain is None:
            return

        # If a snapshot was just taken this frame, run verification now.
        if self._verification_snapshot_done:
            self._verification_snapshot_done = False
            self._verify_pickups()
            return

        if not self._pending_verification or self.robot_pose is None:
            return

        robot_radius = float(self.params.get("robot_radius_cm", 30.0))
        for ball in self._pending_verification:
            dist = (
                (ball.cm_x - self.robot_pose.x_cm) ** 2
                + (ball.cm_y - self.robot_pose.y_cm) ** 2
            ) ** 0.5
            if dist < robot_radius:
                return  # still near at least one pending ball — wait

        # Robot is clear of all pending balls — schedule verification snapshot.
        self._needs_verification_snapshot = True

    def _verify_pickups(self) -> None:
        """Compare fresh YOLO detections against pending pickups; replan if any failed."""
        result = self._last_result
        pending = list(self._pending_verification)
        self._pending_verification = []
        self._attempted_pickups = 0

        if not pending or result is None:
            return

        match_radius_sq = 15.0 ** 2
        fresh = result.smoothed_ball_coordinates
        failed = [
            p for p in pending
            if any(
                (b.cm_x - p.cm_x) ** 2 + (b.cm_y - p.cm_y) ** 2 < match_radius_sq
                for b in fresh
            )
        ]

        log_event("BRAIN", "pickup verification", attempted=len(pending), failed=len(failed))

        if not failed:
            return  # all pickups confirmed

        self.message = f"Pickup failed for {len(failed)} ball(s) — replanning"
        if (
            result.smoothed_ball_coordinates
            and result.occupancy_grid is not None
            and self.robot_pose is not None
            and self._brain is not None
        ):
            self._replan_after_displacement()

    # ------------------------------------------------------------------
    # Red-cross collision re-localization (fire-on-exit)
    # ------------------------------------------------------------------

    def _tick_cross_tracker(self) -> None:
        """Re-localize the manual cross after the robot drives through its hitbox.

        Cheap every frame: a single point-in-polygon test against the cross
        inflated by half the robot width.  The state machine and the red
        re-detection live in ``perception.vision.cross_tracking``; this only
        wires them to the live robot pose, occupancy grid, and route planner.
        """
        if self.mode != AppMode.AUTO or self._brain is None:
            return

        # Deferred replan: the grid was rebuilt with the new cross last frame.
        if self._cross_tracker.take_pending_replan():
            self._replan_after_cross_move()
            return

        if self._cross_spec is None or self.robot_pose is None:
            return

        half_robot = 0.5 * float(self.params.get("robot_width_cm", 20.0))
        in_buffer = self._cross_spec.contains_point(
            (self.robot_pose.x_cm, self.robot_pose.y_cm), half_robot
        )
        if self._cross_tracker.step(in_buffer) is CrossAction.RELOCALIZE:
            self._relocalize_cross()

    def _relocalize_cross(self) -> None:
        """Re-detect the displaced cross near its last pose and update the spec silently."""
        result = self._last_result
        spec = self._cross_spec
        if result is None or spec is None or result.frame_for_detection is None:
            return

        new_spec = relocalize_cross(
            result.frame_for_detection,
            spec,
            self.params,
            self.pipeline.mapper,
            self.pipeline.red_zone_detector,
        )
        if new_spec is None:
            log_event("BRAIN", "cross re-localize: not found, keeping prior pose")
            return

        moved = math.hypot(
            new_spec.center_cm[0] - spec.center_cm[0],
            new_spec.center_cm[1] - spec.center_cm[1],
        )
        self._cross_spec = new_spec
        self._save_cross()
        # Defer the replan one frame so _process_frame rebuilds the occupancy
        # grid with the cross in its new position first.
        self._cross_tracker.request_replan()
        log_event(
            "BRAIN", "cross re-localized after collision",
            moved_cm=round(moved, 1),
            center=(round(new_spec.center_cm[0], 1), round(new_spec.center_cm[1], 1)),
        )

    def _replan_after_cross_move(self) -> None:
        """Replan the active route around the updated cross, reusing known ball targets.

        Unlike ``_replan_after_displacement`` this sources targets from the
        already-tracked balls, because AUTO frames skip ball detection and so
        carry no fresh ``smoothed_ball_coordinates``.
        """
        result = self._last_result
        if result is None:
            return
        balls = self._tracked_balls if self._tracked_balls else list(result.smoothed_ball_coordinates)
        if not self._plan_and_load_route(self._ball_targets(balls)):
            self.message = "Cross moved — no route found"
            log_event("BRAIN", "cross re-localized but no route found")
            return
        self.message = f"Cross moved — replanned ({self._brain.step_count} steps)"
        log_event("BRAIN", "replanned after cross re-localization", steps=self._brain.step_count)

    # ------------------------------------------------------------------
    # Robot self-calibration (spin -> fit center -> align body -> save)
    # ------------------------------------------------------------------

    def _start_calibration(self) -> None:
        """Connect to the robot and begin an auto-spin origin calibration."""
        if self.mode == AppMode.CALIBRATE:
            self.message = "Calibration already running"
            return
        if self._connecting:
            self.message = "Already connecting..."
            return
        if self.pipeline.preprocessor.homography_calibrator.transform_matrix is None:
            self.message = "Set field corners before calibrating the robot"
            return

        self._calib_backup = self.calibration
        self._calib_collector = RobotCalibrationCollector(self.config.robot, self.config.robot_calibration)
        self._calib_runtime = RobotCalibrationRuntime(
            collected_points={marker_id: [] for marker_id in self.config.robot.marker_ids}
        )
        self.mode = AppMode.CALIBRATE
        self._calib_phase = CALIB_CONNECTING
        self._connecting = True
        self.message = "Connecting to robot for calibration..."

        def connect() -> None:
            try:
                if self._commander is None:
                    self._commander = RobotCommander(drive_config=self.config.drive, auto_connect=True)
                self._spin = SpinController(
                    self._commander,
                    turn_speed_pct=self.config.drive.turn_max_speed_pct,
                )
                self._calib_phase = CALIB_ALIGN
                self.message = "Connected. Press Spin to find center, adjust body, then Save."
            except Exception as exc:
                self.message = f"Calibration connect failed: {exc}"
                self._end_calibration(disconnect=True)
            finally:
                self._connecting = False

        threading.Thread(target=connect, daemon=True).start()

    def _start_spin(self) -> None:
        """Begin the auto-spin from the calibration menu (Spin button)."""
        if self.mode != AppMode.CALIBRATE or self._connecting:
            return
        if self._spin is None:
            self.message = "Still connecting — try Spin again in a moment"
            return
        if self._calib_phase == CALIB_SPIN:
            return
        self._calib_runtime = RobotCalibrationRuntime(
            collected_points={marker_id: [] for marker_id in self.config.robot.marker_ids}
        )
        self._spin.start()
        self._calib_phase = CALIB_SPIN
        self.message = "Spinning robot — keep markers visible (sweeping past 360 deg)"

    def _tick_calibration(self) -> None:
        """Drive the auto-spin and point collection while in CALIBRATE/spin."""
        if self.mode != AppMode.CALIBRATE or self._calib_phase != CALIB_SPIN:
            return
        if self._spin is None or self._connecting or self._calib_runtime is None:
            return

        marker_yaws: dict[int, float] = {}
        for marker_id, observation in self._latest_observations.items():
            self._calib_runtime.collected_points.setdefault(marker_id, []).append(
                (float(observation.ground_center[0]), float(observation.ground_center[1]))
            )
            marker_yaws[marker_id] = float(observation.yaw_rad)

        status = self._spin.tick(marker_yaws)
        if status in (SpinStatus.COMPLETE, SpinStatus.TIMED_OUT):
            self._finish_spin(status)

    def _finish_spin(self, status: SpinStatus) -> None:
        """Fit turning centers and enter the body-alignment phase."""
        runtime = self._calib_runtime
        collector = self._calib_collector
        if runtime is None or collector is None:
            return

        if not collector.compute_spin_centers(runtime):
            self.message = f"Spin calibration failed: {runtime.warning}"
            self._cancel_calibration()
            return

        if self._latest_parallax is None or not self._latest_observations:
            self.message = "Lost markers right after spin — show a marker and retry"
            self._cancel_calibration()
            return

        # Provisional (in-memory) calibration so the live pose + overlay work
        # immediately during alignment; only written to disk on Save.
        self.calibration = collector.build_robot_calibration(
            runtime,
            self._latest_observations,
            self._latest_parallax,
            self.config.camera.topdown_warp_size,
            existing=self._calib_backup or {},
        )
        self._calib_phase = CALIB_ALIGN
        swept = self._spin.swept_deg if self._spin else 0.0
        timeout_note = " (timed out)" if status == SpinStatus.TIMED_OUT else ""
        extra = f" — {runtime.warning}" if runtime.warning else ""
        self.message = (
            f"Spin done: {swept:.0f} deg{timeout_note}. Align body with +/- buttons, then Save.{extra}"
        )

    def _geometry_nudge(self, param_key: str, delta: float) -> None:
        """Adjust one live geometry/heading parameter and refresh the summary."""
        if self.mode != AppMode.CALIBRATE or self._calib_phase != CALIB_ALIGN:
            return
        current = float(self.params.get(param_key, 0.0))
        updated = current + delta
        if param_key != "heading_tuning_rad":
            updated = max(0.0, updated)
        self.params[param_key] = updated
        self.message = self._geometry_summary()

    def _geometry_summary(self) -> str:
        p = self.params
        return (
            f"Front {float(p['robot_front_cm']):.1f}  Rear {float(p['robot_rear_cm']):.1f}  "
            f"Width {float(p['robot_width_cm']):.1f}  Pipe {float(p['tube_forward_cm']):.1f} cm  "
            f"Head {math.degrees(float(p['heading_tuning_rad'])):+.1f} deg"
        )

    def _save_calibration(self) -> None:
        """Persist calibration to robot_calibration.json.

        If a spin was run this session, save the new turning-center offsets plus
        geometry. Otherwise save only the geometry/pipe/heading tuning, preserving
        the existing marker offsets on disk (adjust-without-respin).
        """
        if self._calib_collector is None:
            return
        geometry = self.pose_estimator.robot_geometry_from_params(self.params)
        heading_tuning = float(self.params.get("heading_tuning_rad", 0.0))
        path = self.config.paths.robot_calibration_file
        runtime = self._calib_runtime
        has_spin = runtime is not None and bool(runtime.fitted_centers)

        if has_spin:
            if self._latest_parallax is None:
                self.message = "Cannot save: no marker geometry this frame"
                return
            visible = [m for m in runtime.fitted_centers if m in self._latest_observations]
            if not visible:
                self.message = "Cannot save: show a calibrated marker to the camera, then Save"
                return
            self._calib_collector.save_robot_calibration(
                path,
                runtime,
                self._latest_observations,
                self._latest_parallax,
                self.config.camera.topdown_warp_size,
                geometry=geometry,
                heading_tuning_rad=heading_tuning,
            )
            saved_message = f"Saved spin + geometry to {path.name}"
        else:
            self._calib_collector.save_geometry_tuning(path, geometry, heading_tuning)
            saved_message = f"Saved geometry tuning to {path.name}"

        self._end_calibration(disconnect=True)
        self._load_robot_calibration()
        self.message = saved_message

    def _cancel_calibration(self) -> None:
        """Abort calibration and restore the pre-calibration state."""
        self.calibration = self._calib_backup
        self.message = "Calibration cancelled"
        self._end_calibration(disconnect=True)

    def _end_calibration(self, disconnect: bool) -> None:
        """Stop the spin, release calibration state, and return to idle."""
        if self._spin is not None:
            self._spin.stop()
        self._spin = None
        self._calib_runtime = None
        self._calib_collector = None
        self._calib_backup = None
        self._calib_phase = CALIB_IDLE
        if disconnect and self._commander is not None:
            try:
                self._commander.stop(force=True)
                self._commander.close()
            except Exception:
                pass
            self._commander = None
        self.mode = AppMode.IDLE

    # ------------------------------------------------------------------
    # Route View window (pickup geometry + route visualization)
    # ------------------------------------------------------------------

    def _open_route_view(self) -> None:
        """Open or re-focus the Route View secondary window."""
        if self._route_view_open:
            return
        self._route_view_open = True
        self._route_view_cache_id = 0
        self._route_view_cached_image = None

        top = tk.Toplevel(self._root)
        top.title(ROUTE_VIEW_WINDOW_NAME)
        top.resizable(False, False)
        top.protocol("WM_DELETE_WINDOW", self._close_route_view)
        self._route_toplevel = top

        self._route_label = tk.Label(top, bg="#000000")
        self._route_label.pack()
        self._route_photo: ImageTk.PhotoImage | None = None

        # Strategy slider replaces cv2.createTrackbar
        strategy_frame = tk.Frame(top, bg="#323232")
        strategy_frame.pack(fill=tk.X, padx=4, pady=4)
        tk.Label(strategy_frame, text="Strategy:", fg="#FFFFFF", bg="#323232",
                 font=("Helvetica", 10)).pack(side=tk.LEFT)
        self._strategy_scale = tk.Scale(
            strategy_frame, from_=0, to=len(STRATEGY_OPTIONS) - 1,
            orient=tk.HORIZONTAL, bg="#323232", fg="#FFFFFF",
            highlightthickness=0, troughcolor="#5A5A5A",
            command=self._on_strategy_change,
        )
        self._strategy_scale.set(self._route_view_strategy_index)
        self._strategy_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.message = "Route View opened — use Strategy slider to switch"

    def _on_strategy_change(self, value: str) -> None:
        """Callback for the route view strategy Scale widget."""
        idx = int(value)
        if 0 <= idx < len(STRATEGY_OPTIONS):
            if idx != self._route_view_strategy_index:
                self._route_view_strategy_index = idx
                self._sync_route_view_strategy_to_planner()

    def _route_view_strategy_index_from_config(self) -> int:
        """Return the route-view option matching the configured route strategy."""
        configured_key = ROUTE_VIEW_KEY_BY_CONFIG_STRATEGY.get(self.config.planner.route_strategy)
        for index, option in enumerate(STRATEGY_OPTIONS):
            if option.key == configured_key:
                return index
        return 0

    def _sync_route_view_strategy_to_planner(self) -> None:
        """No-op: fit-based pathing uses a single strategy (NearestNeighborStrategy)."""
        pass

    def _close_route_view(self) -> None:
        self._route_view_open = False
        self._route_view_cached_image = None
        if hasattr(self, "_route_toplevel") and self._route_toplevel is not None:
            try:
                self._route_toplevel.destroy()
            except tk.TclError:
                pass
            self._route_toplevel = None

    def _render_route_image(self) -> np.ndarray:
        """Render the route view image and return it as a numpy array."""
        result = self._last_result
        if result is None or result.occupancy_grid is None:
            placeholder = np.full(
                (self._right_h, self._right_w, 3), (40, 40, 40), dtype=np.uint8,
            )
            cv2.putText(
                placeholder, "Waiting for perception data...",
                (20, self._right_h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA,
            )
            return placeholder

        # Cache check: skip recompute if data unchanged and strategy unchanged
        cache_id = id(result)
        pose_id = id(self.robot_pose)
        combined_id = hash((cache_id, pose_id, self._route_view_strategy_index))
        if combined_id == self._route_view_cache_id and self._route_view_cached_image is not None:
            return self._route_view_cached_image
        self._route_view_cache_id = combined_id

        # Build base schematic
        frame_shape = (
            result.preprocessed.topdown.shape
            if result.preprocessed.topdown is not None
            else (self._left_h, self._left_w, 3)
        )
        camera_center = (
            float(self.params.get("camera_center_x", self._left_w / 2)),
            float(self.params.get("camera_center_y", self._left_h / 2)),
        )
        image = self.renderer.draw_schematic(
            frame_shape=frame_shape,
            red_zones=result.red_zones,
            smoothed_ball_coordinates=result.smoothed_ball_coordinates,
            camera_center_pixels=camera_center,
            robot_pose=self.robot_pose,
            params=self.params,
        )

        # Compute pickup geometry
        geometry = self.pose_estimator.robot_geometry_from_params(self.params)
        field_w = self.config.field.width_cm
        field_h = self.config.field.height_cm
        targets = [
            PlannedBallTarget(
                track_id=b.track_id,
                label=b.label,
                x_cm=b.cm_x,
                y_cm=b.cm_y,
                node_cm=self.pipeline.mapper.field_metric_cm_to_grid_node(
                    (b.cm_x, b.cm_y),
                ),
            )
            for b in result.smoothed_ball_coordinates
        ]

        if not targets:
            self._route_view_cached_image = image
            return image

        geometry_result = compute_pickup_geometry(
            field_w, field_h, result.occupancy_grid, targets, geometry,
        )
        mapper = self.renderer.mapper
        draw_pickup_geometry(image, geometry_result, mapper, self.config.field)

        # Build start pose and route
        if self.robot_pose is not None:
            start_pose = HybridPose(
                x_cm=self.robot_pose.x_cm,
                y_cm=self.robot_pose.y_cm,
                theta_rad=self.robot_pose.heading_rad,
            )
        else:
            start_pose = HybridPose(x_cm=20.0, y_cm=20.0, theta_rad=0.0)

        unload_reach = geometry.rear_cm + geometry.unload_extension_cm
        unload_pos = (unload_reach + 2.0, self.config.field.height_cm * 0.5)

        from path.route_strategy import NearestNeighborStrategy, IntersectionPriorityStrategy, RoutePlannerInput
        route_input = RoutePlannerInput(
            geometry_result=geometry_result,
            start_pose=start_pose,
            field_width_cm=field_w,
            field_height_cm=field_h,
            unload_position=unload_pos,
        )
        strategy = IntersectionPriorityStrategy()
        strategy_result = strategy.plan(route_input)
        draw_route_plan(image, strategy_result, geometry_result, mapper, self.config.field)

        self._route_view_cached_image = image
        return image

    # ------------------------------------------------------------------
    # Per-frame processing
    # ------------------------------------------------------------------

    def _process_frame(self, raw_frame: np.ndarray) -> VisionFrameResult:
        # HSV cross detection is disabled; the central cross is the manually
        # placed one, fed in as a red zone so it flows into the occupancy grid
        # and both panel overlays.
        skip = (
            self.mode == AppMode.GUIDANCE_TEST
            or (self.mode == AppMode.AUTO and self._tracked_balls is not None and not self._needs_verification_snapshot)
        )
        if self._needs_verification_snapshot:
            self._needs_verification_snapshot = False
            self._verification_snapshot_done = True
        cross = self.cross_red_zone()
        extra_red_zones = [cross] if cross is not None else []
        return self.pipeline.process(
            raw_frame,
            params=self.params,
            use_aruco=True,
            skip_ball_detection=skip,
            detect_red_zones=False,
            extra_red_zones=extra_red_zones,
        )

    def _estimate_pose(self, result: VisionFrameResult) -> None:
        topdown = result.preprocessed.topdown
        if topdown is None or self.params is None:
            self.robot_pose = None
            self._latest_observations = {}
            self._latest_parallax = None
            return
        pose, _origin_px, observations, parallax = self.pose_estimator.estimate(
            topdown, self.params, self.calibration,
        )
        self.robot_pose = pose
        self._latest_observations = observations
        self._latest_parallax = parallax

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
            left = self.renderer.make_topdown_placeholder("No top-down warp — click Set Corners")

        if self._tracked_balls is not None:
            self._draw_crop_overlay(left)
        if left.shape[1] != self._left_w or left.shape[0] != self._left_h:
            left = cv2.resize(left, (self._left_w, self._left_h), interpolation=cv2.INTER_LINEAR)
        if self.mode == AppMode.CALIBRATE:
            self._draw_calibration_overlay(left)
        return left

    def _draw_crop_overlay(self, image: np.ndarray) -> None:
        """Draw crop monitor boxes on the topdown frame.

        Green = ball present, Red = missing, Grey = skipped (robot over crop).
        """
        if not self._tracked_balls:
            return
        crop_size = int(self.params.get("crop_size", 60))
        robot_radius_cm = float(self.params.get("robot_radius_cm", 30.0))
        half = crop_size // 2
        h, w = image.shape[:2]
        px_per_cm = w / self.pipeline.config.field.width_cm

        robot_px = None
        if self.robot_pose is not None:
            robot_px = self.pipeline.mapper.field_cm_to_topdown_pixel(
                (self.robot_pose.x_cm, self.robot_pose.y_cm)
            )
        robot_radius_px = robot_radius_cm * px_per_cm

        # Draw robot exclusion circle so operator can see which crops are protected
        if robot_px is not None:
            cv2.circle(
                image,
                (int(round(robot_px[0])), int(round(robot_px[1]))),
                int(round(robot_radius_px)),
                (80, 80, 80),
                1,
                cv2.LINE_AA,
            )

        for ball in self._tracked_balls:
            if ball.label != "white":
                continue  # orange balls not monitored via HSV
            cx, cy = self.pipeline.mapper.field_cm_to_topdown_pixel((ball.cm_x, ball.cm_y))
            cx, cy = int(round(cx)), int(round(cy))

            if robot_px is not None:
                dist = ((cx - robot_px[0]) ** 2 + (cy - robot_px[1]) ** 2) ** 0.5
                if dist < robot_radius_px:
                    cv2.rectangle(image, (cx - half, cy - half), (cx + half, cy + half), (120, 120, 120), 1, cv2.LINE_AA)
                    continue

            color = (60, 60, 200) if ball.track_id in self._last_crop_missing else (60, 200, 60)
            cv2.rectangle(image, (cx - half, cy - half), (cx + half, cy + half), color, 2, cv2.LINE_AA)

        # Draw pending verification cluster overlay
        if self._pending_verification:
            pv_px = [
                self.pipeline.mapper.field_cm_to_topdown_pixel((b.cm_x, b.cm_y))
                for b in self._pending_verification
            ]
            # Dashed yellow box per pending ball
            for px, py in pv_px:
                px, py = int(round(px)), int(round(py))
                cv2.rectangle(image, (px - half, py - half), (px + half, py + half), (0, 220, 220), 1, cv2.LINE_AA)
                cv2.drawMarker(image, (px, py), (0, 220, 220), cv2.MARKER_CROSS, 10, 1, cv2.LINE_AA)

            # Cluster center crosshair
            cx_mean = sum(p[0] for p in pv_px) / len(pv_px)
            cy_mean = sum(p[1] for p in pv_px) / len(pv_px)
            center_px = (int(round(cx_mean)), int(round(cy_mean)))

            # Cluster radius = max distance from center to any pending ball
            cluster_r = max(
                int(round(((p[0] - cx_mean) ** 2 + (p[1] - cy_mean) ** 2) ** 0.5))
                for p in pv_px
            ) + half
            cv2.circle(image, center_px, cluster_r, (0, 220, 220), 1, cv2.LINE_AA)
            cv2.drawMarker(image, center_px, (0, 220, 220), cv2.MARKER_TILTED_CROSS, 14, 2, cv2.LINE_AA)

            label = f"Pending {self._attempted_pickups}x"
            cv2.putText(image, label, (center_px[0] + cluster_r + 4, center_px[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 220), 1, cv2.LINE_AA)

    def _draw_calibration_overlay(self, image: np.ndarray) -> None:
        """Overlay the spin turning-centers and the live virtual body on the feed.

        Drawn on the top-down camera panel (left) so the operator can align the
        virtual footprint with the physical robot. Top-down pixel space matches
        ``field_cm_to_topdown_pixel`` output.
        """
        mapper = self.pipeline.mapper

        if self._calib_phase == CALIB_ALIGN and self._calib_runtime is not None:
            for marker_id, center_px in self._calib_runtime.fitted_centers.items():
                cx, cy = int(round(center_px[0])), int(round(center_px[1]))
                cv2.drawMarker(image, (cx, cy), (0, 165, 255), cv2.MARKER_TILTED_CROSS, 18, 2, cv2.LINE_AA)
                cv2.putText(image, f"center {marker_id}", (cx + 10, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1, cv2.LINE_AA)

        if self.robot_pose is not None:
            geometry = self.pose_estimator.robot_geometry_from_params(self.params)
            live_pose = HybridPose(self.robot_pose.x_cm, self.robot_pose.y_cm, self.robot_pose.heading_rad)
            base_cm, intake_cm = self.renderer.robot_footprint_metric_polygons(live_pose, geometry)
            base_px = np.array([mapper.field_cm_to_topdown_pixel(p) for p in base_cm], dtype=np.int32).reshape(-1, 1, 2)
            intake_px = np.array([mapper.field_cm_to_topdown_pixel(p) for p in intake_cm], dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(image, [base_px], True, (255, 90, 30), 2, cv2.LINE_AA)
            cv2.polylines(image, [intake_px], True, (0, 255, 255), 2, cv2.LINE_AA)
            origin = mapper.field_cm_to_topdown_pixel((self.robot_pose.x_cm, self.robot_pose.y_cm))
            tube = mapper.field_cm_to_topdown_pixel((self.robot_pose.tube_x_cm, self.robot_pose.tube_y_cm))
            origin_xy = (int(round(origin[0])), int(round(origin[1])))
            tube_xy = (int(round(tube[0])), int(round(tube[1])))
            cv2.circle(image, origin_xy, 5, (255, 90, 30), -1, cv2.LINE_AA)
            cv2.arrowedLine(image, origin_xy, tube_xy, (0, 255, 255), 2, cv2.LINE_AA, tipLength=0.3)

        if self._calib_phase == CALIB_SPIN and self._spin is not None:
            cv2.putText(image, f"SPINNING {self._spin.swept_deg:.0f}/{self._spin.target_sweep_deg:.0f} deg",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        elif self._calib_phase == CALIB_ALIGN:
            cv2.putText(image, "Spin to find center | match box to robot | Save",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2, cv2.LINE_AA)

    def _build_right_panel(self, result: VisionFrameResult) -> np.ndarray:
        frame_shape = (
            result.preprocessed.topdown.shape
            if result.preprocessed.topdown is not None
            else (self._left_h, self._left_w, 3)
        )
        camera_center = (
            float(self.params.get("camera_center_x", self._left_w / 2)),
            float(self.params.get("camera_center_y", self._left_h / 2)),
        )

        # Pass route waypoints to the schematic when guidance or brain is active
        manual_wp: list[tuple[float, float]] | None = None
        route_pts: list[HybridPose] | None = None
        if self._active_route is not None and self.mode == AppMode.GUIDANCE_TEST:
            manual_wp = [(wp.x_cm, wp.y_cm) for wp in self._active_route]
            route_pts = self._active_route
        elif self._brain_route_points is not None and self.mode == AppMode.AUTO:
            manual_wp = [(wp.x_cm, wp.y_cm) for wp in self._brain_route_points]
            route_pts = self._brain_route_points

        image = self.renderer.draw_schematic(
            frame_shape=frame_shape,
            red_zones=result.red_zones,
            smoothed_ball_coordinates=result.smoothed_ball_coordinates,
            camera_center_pixels=camera_center,
            robot_pose=self.robot_pose,
            params=self.params,
            route_points_cm=route_pts,
            manual_waypoints_cm=manual_wp,
        )

        # Pickup geometry overlay (heatmap / collision map)
        if self._overlay_mode != OverlayMode.NONE and result.occupancy_grid is not None:
            geometry = self.pose_estimator.robot_geometry_from_params(self.params)
            targets = self._ball_targets(result.smoothed_ball_coordinates)
            if targets:
                geometry_result = compute_pickup_geometry(
                    self.config.field.width_cm, self.config.field.height_cm,
                    result.occupancy_grid, targets, geometry,
                )
                half_w = geometry.width_cm * 0.5
                draw_pickup_geometry(
                    image, geometry_result, self.renderer.mapper,
                    self.config.field, self._overlay_mode, half_w,
                )

        return image

    def _read_frame(self) -> np.ndarray | None:
        if self.static_image is not None:
            return self.static_image.copy()
        if self.camera is not None:
            ret, frame = self.camera.read()
            if ret:
                return frame
        return None

    # ------------------------------------------------------------------
    # Vision thread
    # ------------------------------------------------------------------

    def _vision_loop(self) -> None:
        """Daemon thread: capture → process → tick controllers → render panels."""
        while not self.closed:
            frame_start = time.perf_counter()

            # Snapshot params under lock so GUI button nudges are safe
            with self._params_lock:
                params_snapshot = dict(self.params) if self.params else {}
            self.params = params_snapshot

            raw_frame = self._read_frame()

            # Render secondary window images (pure rendering, no cv2.imshow)
            corner_img = None
            cross_img = None
            route_img = None
            if self._corner_window_open and raw_frame is not None:
                corner_img = self._render_corner_image(raw_frame)
            if self._cross_window_open and raw_frame is not None:
                cross_img = self._render_cross_image(raw_frame)
            if self._route_view_open:
                route_img = self._render_route_image()

            if raw_frame is not None:
                result = self._process_frame(raw_frame)
                self._last_result = result
                self._estimate_pose(result)
                self._tick_guidance()
                self._tick_crop_monitor()
                self._tick_pickup_verifier()
                self._tick_cross_tracker()
                self._tick_brain()
                self._tick_calibration()
                left = self._build_left_panel(result)
                right = self._build_right_panel(result)
            else:
                left = self.renderer.make_topdown_placeholder("No camera input")
                if left.shape[1] != self._left_w or left.shape[0] != self._left_h:
                    left = cv2.resize(left, (self._left_w, self._left_h))
                right = np.full((self._right_h, self._right_w, 3), (40, 100, 40), dtype=np.uint8)
                cv2.putText(right, "No camera input", (30, self._right_h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)

            dt = time.perf_counter() - frame_start
            self.fps = 1.0 / max(dt, 1e-6)

            # Publish to shared state under lock
            self._frame_seq += 1
            with self._display_lock:
                self._shared.left_panel = left
                self._shared.right_panel = right
                self._shared.corner_image = corner_img
                self._shared.cross_image = cross_img
                self._shared.route_image = route_img
                self._shared.frame_seq = self._frame_seq
                self._shared.fps = self.fps
                self._shared.mode = self.mode.value
                self._shared.message = self.message
                self._shared.robot_pose = self.robot_pose
                self._shared.brain_state = self._brain_state.value if self._brain_state else None
                self._shared.guidance_status = self._guidance_status.value if self._guidance_status else None
                self._shared.calibration_phase = self._calib_phase
                self._shared.cal_state = self.pipeline.preprocessor.homography_calibrator.calibration_state.value
                self._shared.connecting = self._connecting
                self._shared.commander_connected = bool(self._commander and self._commander.sock)
                self._shared.active_route_name = self._active_route_name
                self._shared.guidance_cursor = self._guidance.cursor if self._guidance else 0
                self._shared.guidance_wp_count = self._guidance.waypoint_count if self._guidance else 0
                self._shared.brain_step_cursor = self._brain.step_cursor if self._brain else 0
                self._shared.brain_step_count = self._brain.step_count if self._brain else 0
                self._shared.timer_elapsed = self._timer_elapsed
                self._shared.markers_visible = ",".join(str(m) for m in sorted(self._latest_observations)) or "none"
                self._shared.corner_window_open = self._corner_window_open
                self._shared.cross_window_open = self._cross_window_open
                self._shared.route_view_open = self._route_view_open

    def _params_write(self, key: str, value) -> None:
        """Thread-safe write to a single params key (used by GUI thread button nudges)."""
        with self._params_lock:
            self.params[key] = value

    def _params_read(self, key: str, default=None):
        """Thread-safe read of a single params key."""
        with self._params_lock:
            return self.params.get(key, default)

    # ------------------------------------------------------------------
    # Tkinter UI construction (main thread)
    # ------------------------------------------------------------------

    def _build_tk_ui(self) -> None:
        """Build the tkinter window and all widgets."""
        self._root = tk.Tk()
        self._root.title(WINDOW_NAME)
        self._root.configure(bg="#323232")
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._root.bind("<Key>", self._on_key)

        # PhotoImage references (prevent GC)
        self._left_photo: ImageTk.PhotoImage | None = None
        self._right_photo: ImageTk.PhotoImage | None = None

        # --- Panel frame: left (camera) + right (schematic) ---
        panel_frame = tk.Frame(self._root, bg="#000000")
        panel_frame.pack(side=tk.TOP, fill=tk.BOTH)

        self._left_label = tk.Label(panel_frame, bg="#000000")
        self._left_label.pack(side=tk.LEFT, padx=0, pady=0)

        self._right_label = tk.Label(panel_frame, bg="#000000")
        self._right_label.pack(side=tk.LEFT, padx=0, pady=0)

        # --- Status frame ---
        status_frame = tk.Frame(self._root, bg="#323232")
        status_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(4, 0))

        self._status_line1 = tk.Label(
            status_frame, text="", anchor=tk.W,
            font=("Courier", 11), fg="#DCDCDC", bg="#323232",
        )
        self._status_line1.pack(fill=tk.X)

        self._status_line2 = tk.Label(
            status_frame, text="", anchor=tk.W,
            font=("Courier", 11), fg="#B4DCFF", bg="#323232",
        )
        self._status_line2.pack(fill=tk.X)

        self._message_label = tk.Label(
            status_frame, text="Ready", anchor=tk.W,
            font=("Courier", 11), fg="#B4DCB4", bg="#323232",
        )
        self._message_label.pack(fill=tk.X)

        self._keys_label = tk.Label(
            status_frame, anchor=tk.W,
            text="Keys: q/Esc quit | f corners | x cross | c calib | g guide | v overlay | a auto | s stop",
            font=("Courier", 9), fg="#A0A0A0", bg="#323232",
        )
        self._keys_label.pack(fill=tk.X)

        # --- Button frame ---
        # Using tk.Label widgets instead of tk.Button because macOS Aqua
        # rendering ignores bg/fg on native buttons, making them unreadable.
        button_frame = tk.Frame(self._root, bg="#323232")
        button_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(4, 4))

        button_specs = [
            ("Set Corners", "set_corners"),
            ("Set Cross", "set_cross"),
            ("Calib Robot", "calib_robot"),
            ("Guide Test", "guidance_test"),
            ("Route View", "route_view"),
            ("Manual", "manual"),
            ("Auto", "auto"),
            ("Stop", "stop"),
            ("Quit", "quit"),
        ]
        self._tk_buttons: dict[str, tk.Label] = {}
        for label, action in button_specs:
            btn = tk.Label(
                button_frame, text=label, width=10,
                relief=tk.RAISED, bg="#5A5A5A", fg="#FFFFFF",
                font=("Helvetica", 10), padx=4, pady=4, cursor="hand2",
            )
            btn.pack(side=tk.LEFT, padx=2, pady=2)
            btn.bind("<Button-1>", lambda e, a=action: self._handle_button(a))
            self._tk_buttons[action] = btn

        # --- Calibration sub-buttons (hidden by default) ---
        self._calib_frame = tk.Frame(self._root, bg="#323232")
        # Not packed initially — shown/hidden dynamically

        calib_specs = [
            ("Spin", "calib_spin"),
            ("Front+", "front_inc"), ("Front-", "front_dec"),
            ("Rear+", "rear_inc"), ("Rear-", "rear_dec"),
            ("Width+", "width_inc"), ("Width-", "width_dec"),
            ("Pipe+", "pipe_inc"), ("Pipe-", "pipe_dec"),
            ("Head+", "head_inc"), ("Head-", "head_dec"),
            ("Save", "calib_save"), ("Cancel", "calib_cancel"),
        ]
        self._calib_buttons: dict[str, tk.Label] = {}
        for label, action in calib_specs:
            btn = tk.Label(
                self._calib_frame, text=label, width=7,
                relief=tk.RAISED, bg="#5A5A5A", fg="#FFFFFF",
                font=("Helvetica", 9), padx=2, pady=3, cursor="hand2",
            )
            btn.pack(side=tk.LEFT, padx=1, pady=2)
            btn.bind("<Button-1>", lambda e, a=action: self._handle_button(a))
            self._calib_buttons[action] = btn

        self._calib_frame_visible = False
        self._last_gui_seq = -1

    def _bgr_to_photo(self, bgr: np.ndarray) -> ImageTk.PhotoImage:
        """Convert a BGR numpy array to a tkinter PhotoImage."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return ImageTk.PhotoImage(image=Image.fromarray(rgb))

    # ------------------------------------------------------------------
    # GUI tick (main thread, called via root.after)
    # ------------------------------------------------------------------

    def _gui_tick(self) -> None:
        """Poll shared state and update tkinter widgets (~30 fps)."""
        if self.closed:
            self._root.quit()
            return

        # Read shared state snapshot under lock
        with self._display_lock:
            left = self._shared.left_panel
            right = self._shared.right_panel
            seq = self._shared.frame_seq
            s_mode = self._shared.mode
            s_message = self._shared.message
            s_fps = self._shared.fps
            s_robot_pose = self._shared.robot_pose
            s_brain_state = self._shared.brain_state
            s_guidance_status = self._shared.guidance_status
            s_calib_phase = self._shared.calibration_phase
            s_cal_state = self._shared.cal_state
            s_connecting = self._shared.connecting
            s_commander_connected = self._shared.commander_connected
            s_active_route_name = self._shared.active_route_name
            s_guidance_cursor = self._shared.guidance_cursor
            s_guidance_wp_count = self._shared.guidance_wp_count
            s_brain_step_cursor = self._shared.brain_step_cursor
            s_brain_step_count = self._shared.brain_step_count
            s_timer_elapsed = self._shared.timer_elapsed
            s_markers_visible = self._shared.markers_visible
            corner_img = self._shared.corner_image
            cross_img = self._shared.cross_image
            route_img = self._shared.route_image

        # Update panel images if new frame available
        if left is not None and right is not None and seq != self._last_gui_seq:
            self._last_gui_seq = seq

            self._left_photo = self._bgr_to_photo(left)
            self._left_label.configure(image=self._left_photo)

            self._right_photo = self._bgr_to_photo(right)
            self._right_label.configure(image=self._right_photo)

        # Update secondary window images
        if self._corner_window_open and corner_img is not None and hasattr(self, "_corner_canvas"):
            self._corner_photo = self._bgr_to_photo(corner_img)
            h, w = corner_img.shape[:2]
            self._corner_canvas.configure(width=w, height=h)
            self._corner_canvas.delete("all")
            self._corner_canvas.create_image(0, 0, anchor=tk.NW, image=self._corner_photo)

        if self._cross_window_open and cross_img is not None and hasattr(self, "_cross_canvas"):
            self._cross_photo = self._bgr_to_photo(cross_img)
            h, w = cross_img.shape[:2]
            self._cross_canvas.configure(width=w, height=h)
            self._cross_canvas.delete("all")
            self._cross_canvas.create_image(0, 0, anchor=tk.NW, image=self._cross_photo)

        if self._route_view_open and route_img is not None and hasattr(self, "_route_label"):
            self._route_photo = self._bgr_to_photo(route_img)
            self._route_label.configure(image=self._route_photo)

        # Update status labels
        pose_text = "N/A"
        if s_robot_pose is not None:
            p = s_robot_pose
            pose_text = f"({p.x_cm:.1f}, {p.y_cm:.1f}) heading {math.degrees(p.heading_rad):.1f} deg"
        self._status_line1.configure(
            text=f"Cal: {s_cal_state} | Pose: {pose_text} | Mode: {s_mode} | FPS: {s_fps:.1f}"
        )

        connected = "Connected" if s_commander_connected else "Disconnected"
        if s_connecting:
            connected = "Connecting..."

        line2 = ""
        line2_fg = "#B4DCFF"
        if s_mode == AppMode.GUIDANCE_TEST.value:
            gs = s_guidance_status or "\u2014"
            line2 = f"Guidance: {gs} | WP: {s_guidance_cursor}/{s_guidance_wp_count} | Route: {s_active_route_name} | {connected}"
        elif s_mode == AppMode.AUTO.value:
            bs = s_brain_state or "\u2014"
            timer_str = f"Time: {s_timer_elapsed:.1f}s"
            line2 = f"Brain: {bs} | Step: {s_brain_step_cursor}/{s_brain_step_count} | {connected} | {timer_str}"
            line2_fg = "#B4FFDC"
        elif s_mode == AppMode.CALIBRATE.value:
            line2 = f"Calibrate: phase={s_calib_phase} | markers={s_markers_visible} | {connected}"
        self._status_line2.configure(text=line2, fg=line2_fg)
        self._message_label.configure(text=s_message)

        # Highlight active buttons
        active_actions = set()
        mode_enum = AppMode(s_mode) if s_mode in AppMode.__members__ else AppMode.IDLE
        if self._corner_window_open:
            active_actions.add("set_corners")
        if self._cross_window_open:
            active_actions.add("set_cross")
        if self._route_view_open:
            active_actions.add("route_view")
        if mode_enum == AppMode.CALIBRATE:
            active_actions.add("calib_robot")
        if mode_enum == AppMode.GUIDANCE_TEST:
            active_actions.add("guidance_test")
        if mode_enum == AppMode.MANUAL:
            active_actions.add("manual")
        if mode_enum == AppMode.AUTO:
            active_actions.add("auto")
        if mode_enum == AppMode.IDLE:
            active_actions.add("stop")

        for action, btn in self._tk_buttons.items():
            if action in active_actions:
                btn.configure(bg="#50A050", relief=tk.SUNKEN)
            else:
                btn.configure(bg="#5A5A5A", relief=tk.RAISED)

        # Show/hide calibration sub-buttons
        show_calib = (mode_enum == AppMode.CALIBRATE and s_calib_phase in (CALIB_ALIGN, CALIB_SPIN))
        if show_calib and not self._calib_frame_visible:
            self._calib_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0, 4))
            self._calib_frame_visible = True
        elif not show_calib and self._calib_frame_visible:
            self._calib_frame.pack_forget()
            self._calib_frame_visible = False

        # Schedule next tick
        self._root.after(33, self._gui_tick)

    # ------------------------------------------------------------------
    # Keyboard handling (tkinter)
    # ------------------------------------------------------------------

    def _on_key(self, event: tk.Event) -> None:
        """Handle keyboard events from tkinter."""
        ch = event.char
        keysym = event.keysym

        if keysym == "Escape" or ch == "q":
            if self._corner_window_open:
                self._close_corner_window()
                self.message = "Corner selection cancelled"
            elif self._cross_window_open:
                self._close_cross_window()
                self.message = "Cross placement cancelled"
            else:
                self.closed = True
        elif ch == "f":
            self._handle_button("set_corners")
        elif ch == "g":
            self._handle_button("guidance_test")
        elif ch == "r":
            if self._corner_window_open:
                self._corner_state.points.clear()
            elif self._cross_window_open:
                self._cross_state.points.clear()
                self._cross_state.done = False
        elif ch == "m":
            self._handle_button("manual")
        elif ch == "a":
            self._handle_button("auto")
        elif ch == "c":
            self._handle_button("calib_robot")
        elif ch == "x":
            self._handle_button("set_cross")
        elif ch == "v":
            self._cycle_overlay_mode()
        elif ch == "s":
            self._handle_button("stop")

    _OVERLAY_CYCLE = [OverlayMode.NONE, OverlayMode.HEATMAP, OverlayMode.COLLISION]

    def _cycle_overlay_mode(self) -> None:
        """Cycle the right-panel overlay: off → heatmap → collision → off."""
        idx = self._OVERLAY_CYCLE.index(self._overlay_mode)
        self._overlay_mode = self._OVERLAY_CYCLE[(idx + 1) % len(self._OVERLAY_CYCLE)]
        labels = {OverlayMode.NONE: "off", OverlayMode.HEATMAP: "heatmap", OverlayMode.COLLISION: "collision"}
        self.message = f"Overlay: {labels[self._overlay_mode]}"

    def _on_close(self) -> None:
        """Handle window close button (X)."""
        self.closed = True

    # ------------------------------------------------------------------
    # run() — tkinter main loop + vision thread
    # ------------------------------------------------------------------

    def run(self) -> None:
        self._build_tk_ui()

        # Start vision thread
        vision_thread = threading.Thread(target=self._vision_loop, daemon=True)
        vision_thread.start()

        self._root.after(33, self._gui_tick)
        self._root.mainloop()  # blocks here
        self._cleanup()

    def _cleanup(self) -> None:
        """Release resources on shutdown."""
        if self.mode == AppMode.CALIBRATE:
            self._cancel_calibration()
        self._disconnect_guidance()
        if self._corner_window_open:
            self._close_corner_window()
        if self._cross_window_open:
            self._close_cross_window()
        if self._route_view_open:
            self._close_route_view()
        if self.camera is not None:
            self.camera.release()

    def close(self) -> None:
        if self.mode == AppMode.CALIBRATE:
            self._cancel_calibration()
        self._disconnect_guidance()
        self.closed = True
