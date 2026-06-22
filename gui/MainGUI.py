"""GolfBot Main GUI — camera + 2D schematic with mode controls.

Single-window OpenCV GUI: live top-down camera feed (left) beside the 2D
schematic field view (right), with a status bar showing mode, pose, and FPS.
Guidance isolation testing (Stage 2) is available via the Guide Test button.
"""

from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np

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
    STRATEGY_OPTIONS,
    draw_pickup_geometry,
    draw_route_plan,
)
from config import AppConfig, RouteStrategyName
from perception.vision.debug import DebugRenderer
from perception.vision.models import CalibrationState, RedCrossSpec, RedZoneDetection
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
STATUS_BAR_HEIGHT = 170


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


@dataclass(frozen=True)
class GuiButton:
    label: str
    action: str
    rect: tuple[int, int, int, int]  # x, y, w, h


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


def _corner_on_mouse(event: int, x: int, y: int, _flags: int, state: CornerSelectionState) -> None:
    w, h = state.frame_size
    if w > 0 and h > 0:
        state.cursor = (int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1)))

    if event == cv2.EVENT_RBUTTONDOWN:
        state.points.clear()
        return

    if event == cv2.EVENT_LBUTTONDOWN and len(state.points) < 4:
        state.points.append(state.cursor)
        if len(state.points) == 4:
            state.done = True


def _cross_on_mouse(event: int, x: int, y: int, _flags: int, state: CornerSelectionState) -> None:
    """Collect up to 2 corner clicks for the red-cross placement window."""
    w, h = state.frame_size
    if w > 0 and h > 0:
        state.cursor = (int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1)))

    if event == cv2.EVENT_RBUTTONDOWN:
        state.points.clear()
        state.done = False
        return

    if event == cv2.EVENT_LBUTTONDOWN and len(state.points) < 2:
        state.points.append(state.cursor)
        if len(state.points) == 2:
            state.done = True


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
# Main GUI
# ---------------------------------------------------------------------------

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
        cv2.namedWindow(CROSS_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(CROSS_WINDOW_NAME, _cross_on_mouse, self._cross_state)
        self.message = "Cross window: click an arm TIP corner, then its inner ARMPIT corner"

    def _close_cross_window(self) -> None:
        self._cross_window_open = False
        try:
            cv2.destroyWindow(CROSS_WINDOW_NAME)
        except cv2.error:
            pass

    def _tick_cross_window(self, raw_frame: np.ndarray) -> None:
        """Drive the zoomed cross-placement window for one frame."""
        if not self._cross_window_open:
            return

        try:
            if cv2.getWindowProperty(CROSS_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                self._close_cross_window()
                self.message = "Cross placement cancelled"
                return
        except cv2.error:
            self._close_cross_window()
            self.message = "Cross placement cancelled"
            return

        # Show the top-down (warped) view, where the cross is rectified and pixels
        # map linearly to field cm.
        undistorted = self.pipeline.preprocessor.undistort(raw_frame)
        topdown = self.pipeline.preprocessor.homography_calibrator.warp(undistorted)
        if topdown is None:
            self.message = "No top-down warp — set field corners first"
            self._close_cross_window()
            return

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
        cv2.imshow(CROSS_WINDOW_NAME, view)

        if state.done:
            tip_cm = self.pipeline.mapper.topdown_px_to_field_cm(state.points[0])
            armpit_cm = self.pipeline.mapper.topdown_px_to_field_cm(state.points[1])
            self._cross_spec = RedCrossSpec.from_tip_and_armpit(tip_cm, armpit_cm)
            self._save_cross()
            self._close_cross_window()

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

    def _canvas_width(self) -> int:
        return self._left_w + self._right_w

    def _canvas_height(self) -> int:
        return self._panel_height() + STATUS_BAR_HEIGHT

    def _panel_height(self) -> int:
        return max(self._left_h, self._right_h)

    def buttons(self) -> list[GuiButton]:
        # Buttons sit on a row below the status text lines
        y = self._panel_height() + 90
        bw, bh = 95, 30
        gap = 8
        x0 = 20
        rows = [
            GuiButton("Set Corners", "set_corners", (x0, y, bw, bh)),
            GuiButton("Set Cross", "set_cross", (x0 + (bw + gap), y, bw, bh)),
            GuiButton("Calib Robot", "calib_robot", (x0 + 2 * (bw + gap), y, bw, bh)),
            GuiButton("Guide Test", "guidance_test", (x0 + 3 * (bw + gap), y, bw, bh)),
            GuiButton("Route View", "route_view", (x0 + 4 * (bw + gap), y, bw, bh)),
            GuiButton("Manual", "manual", (x0 + 5 * (bw + gap), y, bw, bh)),
            GuiButton("Auto", "auto", (x0 + 6 * (bw + gap), y, bw, bh)),
            GuiButton("Stop", "stop", (x0 + 7 * (bw + gap), y, bw, bh)),
            GuiButton("Quit", "quit", (x0 + 8 * (bw + gap), y, bw, bh)),
        ]
        # Second row: calibration controls (depends on sub-phase)
        if self.mode == AppMode.CALIBRATE:
            if self._calib_phase == CALIB_ALIGN:
                rows.extend(self._calibration_menu_buttons(y + bh + gap))
            elif self._calib_phase == CALIB_SPIN:
                rows.append(self._calib_button(0, "Cancel", "calib_cancel", y + bh + gap))
        return rows

    @staticmethod
    def _calib_button(index: int, label: str, action: str, y: int) -> GuiButton:
        bw, gap, x0 = 64, 6, 20
        return GuiButton(label, action, (x0 + index * (bw + gap), y, bw, 28))

    def _calibration_menu_buttons(self, y: int) -> list[GuiButton]:
        specs = [
            ("Spin", "calib_spin"),
            ("Front+", "front_inc"), ("Front-", "front_dec"),
            ("Rear+", "rear_inc"), ("Rear-", "rear_dec"),
            ("Width+", "width_inc"), ("Width-", "width_dec"),
            ("Pipe+", "pipe_inc"), ("Pipe-", "pipe_dec"),
            ("Head+", "head_inc"), ("Head-", "head_dec"),
            ("Save", "calib_save"), ("Cancel", "calib_cancel"),
        ]
        return [
            self._calib_button(i, label, action, y)
            for i, (label, action) in enumerate(specs)
        ]

    def handle_mouse(self, event: int, x: int, y: int, _flags: int, _userdata) -> None:
        self._mouse_pos = (x, y)
        if event != cv2.EVENT_LBUTTONUP:
            return
        for button in self.buttons():
            bx, by, bw, bh = button.rect
            if bx <= x <= bx + bw and by <= y <= by + bh:
                self._handle_button(button.action)
                return

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
        cv2.namedWindow(CORNER_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(CORNER_WINDOW_NAME, _corner_on_mouse, self._corner_state)
        self.message = "Corner selection window opened — click 4 inner field corners"

    def _close_corner_window(self) -> None:
        self._corner_window_open = False
        try:
            cv2.destroyWindow(CORNER_WINDOW_NAME)
        except cv2.error:
            pass

    def _tick_corner_window(self, raw_frame: np.ndarray) -> None:
        """Drive the corner-selection window for one frame."""
        if not self._corner_window_open:
            return

        # Check if window was closed by user
        try:
            if cv2.getWindowProperty(CORNER_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                self._close_corner_window()
                self.message = "Corner selection cancelled"
                return
        except cv2.error:
            self._close_corner_window()
            self.message = "Corner selection cancelled"
            return

        # Undistort (but do NOT warp) so the user sees the real camera with
        # barrel distortion removed — the same view the homography is built on.
        undistorted = self.pipeline.preprocessor.undistort(raw_frame)
        self._corner_state.frame_size = (undistorted.shape[1], undistorted.shape[0])
        if self._corner_state.cursor == (0, 0):
            self._corner_state.cursor = (undistorted.shape[1] // 2, undistorted.shape[0] // 2)

        view = _draw_corner_overlay(undistorted, self._corner_state)
        cv2.imshow(CORNER_WINDOW_NAME, view)

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

        if (
            self._brain_state == BrainState.ERROR
            and self._brain.error_message == "ball_displaced"
            and self._last_result is not None
            and self._last_result.smoothed_ball_coordinates
            and self._last_result.occupancy_grid is not None
            and self.robot_pose is not None
        ):
            self._replan_after_displacement()

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

    def _replan_after_displacement(self) -> None:
        """Replan route from the current snapshot after a ball-displaced error."""
        result = self._last_result
        targets = [
            PlannedBallTarget(
                track_id=b.track_id,
                label=b.label,
                x_cm=b.cm_x,
                y_cm=b.cm_y,
                node_cm=self.pipeline.mapper.field_metric_cm_to_grid_node((b.cm_x, b.cm_y)),
            )
            for b in result.smoothed_ball_coordinates
        ]
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
        )

        if not plan.waypoints:
            self.message = "Rescan: no route found — stopping"
            return

        self._brain.load_route(plan)
        self._brain_route_points = [
            HybridPose(w.x_cm, w.y_cm, w.theta_rad) for w in plan.waypoints
        ]
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
        cv2.namedWindow(ROUTE_VIEW_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.createTrackbar(
            "Strategy", ROUTE_VIEW_WINDOW_NAME,
            self._route_view_strategy_index, len(STRATEGY_OPTIONS) - 1,
            lambda _v: None,
        )
        cv2.resizeWindow(
            ROUTE_VIEW_WINDOW_NAME,
            self._right_w, self._right_h,
        )
        self.message = "Route View opened — use Strategy trackbar to switch"

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
        try:
            cv2.destroyWindow(ROUTE_VIEW_WINDOW_NAME)
        except cv2.error:
            pass

    def _tick_route_view(self) -> None:
        """Render one frame of the Route View window."""
        if not self._route_view_open:
            return

        # Check if window was closed by user
        try:
            if cv2.getWindowProperty(ROUTE_VIEW_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                self._close_route_view()
                return
        except cv2.error:
            self._close_route_view()
            return

        # Read trackbar
        tb = cv2.getTrackbarPos("Strategy", ROUTE_VIEW_WINDOW_NAME)
        if 0 <= tb < len(STRATEGY_OPTIONS):
            strategy_changed = tb != self._route_view_strategy_index
            self._route_view_strategy_index = tb
            if strategy_changed:
                self._sync_route_view_strategy_to_planner()
        else:
            strategy_changed = False

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
            cv2.imshow(ROUTE_VIEW_WINDOW_NAME, placeholder)
            return

        # Cache check: skip recompute if data unchanged and strategy unchanged
        cache_id = id(result)
        pose_id = id(self.robot_pose)
        combined_id = hash((cache_id, pose_id, self._route_view_strategy_index))
        if combined_id == self._route_view_cache_id and self._route_view_cached_image is not None:
            cv2.imshow(ROUTE_VIEW_WINDOW_NAME, self._route_view_cached_image)
            return
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
            cv2.imshow(ROUTE_VIEW_WINDOW_NAME, image)
            return

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

        from path.route_strategy import NearestNeighborStrategy, RoutePlannerInput
        route_input = RoutePlannerInput(
            geometry_result=geometry_result,
            start_pose=start_pose,
            field_width_cm=field_w,
            field_height_cm=field_h,
            unload_position=unload_pos,
        )
        strategy = NearestNeighborStrategy()
        strategy_result = strategy.plan(route_input)
        draw_route_plan(image, strategy_result, geometry_result, mapper, self.config.field)

        self._route_view_cached_image = image
        cv2.imshow(ROUTE_VIEW_WINDOW_NAME, image)

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

        return self.renderer.draw_schematic(
            frame_shape=frame_shape,
            red_zones=result.red_zones,
            smoothed_ball_coordinates=result.smoothed_ball_coordinates,
            camera_center_pixels=camera_center,
            robot_pose=self.robot_pose,
            params=self.params,
            route_points_cm=route_pts,
            manual_waypoints_cm=manual_wp,
        )

    def _draw_status_bar(self, canvas: np.ndarray) -> None:
        y0 = self._panel_height()
        cv2.rectangle(canvas, (0, y0), (canvas.shape[1], canvas.shape[0]), (50, 50, 50), -1)

        cal_state = self.pipeline.preprocessor.homography_calibrator.calibration_state.value
        pose_text = "N/A"
        if self.robot_pose is not None:
            p = self.robot_pose
            pose_text = f"({p.x_cm:.1f}, {p.y_cm:.1f}) heading {math.degrees(p.heading_rad):.1f} deg"

        status = f"Cal: {cal_state} | Pose: {pose_text} | Mode: {self.mode.value} | FPS: {self.fps:.1f}"
        cv2.putText(canvas, status, (20, y0 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

        # Guidance status line (line 2) — only when in guidance test mode
        if self.mode == AppMode.GUIDANCE_TEST:
            gs = self._guidance_status.value if self._guidance_status else "—"
            cursor = self._guidance.cursor if self._guidance else 0
            wp_count = self._guidance.waypoint_count if self._guidance else 0
            connected = "Connected" if self._commander and self._commander.sock else "Disconnected"
            if self._connecting:
                connected = "Connecting..."
            guidance_line = f"Guidance: {gs} | WP: {cursor}/{wp_count} | Route: {self._active_route_name} | {connected}"
            cv2.putText(canvas, guidance_line, (20, y0 + 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 220, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, self.message, (20, y0 + 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 220, 180), 1, cv2.LINE_AA)
        elif self.mode == AppMode.AUTO:
            bs = self._brain_state.value if self._brain_state else "—"
            step_cur = self._brain.step_cursor if self._brain else 0
            step_tot = self._brain.step_count if self._brain else 0
            connected = "Connected" if self._commander and self._commander.sock else "Disconnected"
            if self._connecting:
                connected = "Connecting..."
            timer_str = f"Time: {self._timer_elapsed:.1f}s"
            brain_line = f"Brain: {bs} | Step: {step_cur}/{step_tot} | {connected} | {timer_str}"
            cv2.putText(canvas, brain_line, (20, y0 + 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 255, 220), 1, cv2.LINE_AA)
            cv2.putText(canvas, self.message, (20, y0 + 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 220, 180), 1, cv2.LINE_AA)
        elif self.mode == AppMode.CALIBRATE:
            connected = "Connected" if self._commander and self._commander.sock else "Disconnected"
            if self._connecting:
                connected = "Connecting..."
            markers = ",".join(str(m) for m in sorted(self._latest_observations)) or "none"
            calib_line = f"Calibrate: phase={self._calib_phase} | markers={markers} | {connected}"
            cv2.putText(canvas, calib_line, (20, y0 + 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 220, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, self.message, (20, y0 + 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 220, 180), 1, cv2.LINE_AA)
        else:
            cv2.putText(canvas, self.message, (20, y0 + 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 220, 180), 1, cv2.LINE_AA)

        cv2.putText(canvas,
                    "Keys: q/Esc quit | f corners | x cross | c calib | g guide | v route | a auto | s stop",
                    (20, y0 + 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1, cv2.LINE_AA)

        # Buttons drawn on their own row below text (y computed in buttons())
        for button in self.buttons():
            bx, by, bw, bh = button.rect
            is_active = (
                (button.action == "set_corners" and self._corner_window_open)
                or (button.action == "set_cross" and self._cross_window_open)
                or (button.action == "route_view" and self._route_view_open)
                or (button.action == "calib_robot" and self.mode == AppMode.CALIBRATE)
                or (button.action == "guidance_test" and self.mode == AppMode.GUIDANCE_TEST)
                or (button.action == "manual" and self.mode == AppMode.MANUAL)
                or (button.action == "auto" and self.mode == AppMode.AUTO)
                or (button.action == "stop" and self.mode == AppMode.IDLE)
            )
            fill_color = (80, 160, 80) if is_active else (90, 90, 90)
            cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), fill_color, -1, cv2.LINE_AA)
            cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (200, 200, 200), 1, cv2.LINE_AA)
            (tw, _), _ = cv2.getTextSize(button.label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            tx = bx + (bw - tw) // 2
            cv2.putText(canvas, button.label, (tx, by + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

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

        # Drive the corner-selection window if open
        if self._corner_window_open and raw_frame is not None:
            self._tick_corner_window(raw_frame)

        # Drive the cross-placement window if open
        if self._cross_window_open and raw_frame is not None:
            self._tick_cross_window(raw_frame)

        # Drive the route view window if open
        if self._route_view_open:
            self._tick_route_view()

        if raw_frame is not None:
            result = self._process_frame(raw_frame)
            self._last_result = result
            self._estimate_pose(result)
            self._tick_guidance()
            self._tick_crop_monitor()
            self._tick_pickup_verifier()
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

        # Top-align: pad the shorter panel at the bottom with black
        target_h = max(left.shape[0], right.shape[0])
        if left.shape[0] < target_h:
            pad = np.zeros((target_h - left.shape[0], left.shape[1], 3), dtype=np.uint8)
            left = np.vstack([left, pad])
        if right.shape[0] < target_h:
            pad = np.zeros((target_h - right.shape[0], right.shape[1], 3), dtype=np.uint8)
            right = np.vstack([right, pad])

        panels = np.hstack([left, right])
        canvas = np.zeros((panels.shape[0] + STATUS_BAR_HEIGHT, panels.shape[1], 3), dtype=np.uint8)
        canvas[:panels.shape[0], :panels.shape[1]] = panels
        self._draw_status_bar(canvas)

        dt = time.perf_counter() - frame_start
        self.fps = 1.0 / max(dt, 1e-6)
        return canvas

    def handle_key(self, key: int) -> None:
        if key in (27, ord("q")):
            if self._corner_window_open:
                self._close_corner_window()
                self.message = "Corner selection cancelled"
            elif self._cross_window_open:
                self._close_cross_window()
                self.message = "Cross placement cancelled"
            else:
                self.closed = True
        elif key == ord("f"):
            self._handle_button("set_corners")
        elif key == ord("g"):
            self._handle_button("guidance_test")
        elif key == ord("r") and self._corner_window_open:
            self._corner_state.points.clear()
        elif key == ord("r") and self._cross_window_open:
            self._cross_state.points.clear()
            self._cross_state.done = False
        elif key == ord("m"):
            self._handle_button("manual")
        elif key == ord("a"):
            self._handle_button("auto")
        elif key == ord("c"):
            self._handle_button("calib_robot")
        elif key == ord("x"):
            self._handle_button("set_cross")
        elif key == ord("v"):
            self._handle_button("route_view")
        elif key == ord("s"):
            self._handle_button("stop")

    def run(self) -> None:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, self._canvas_width(), self._canvas_height())
        cv2.setMouseCallback(WINDOW_NAME, self.handle_mouse)

        while not self.closed:
            canvas = self.tick()
            cv2.imshow(WINDOW_NAME, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key != 255:
                self.handle_key(key)
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                self.closed = True

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
        cv2.destroyWindow(WINDOW_NAME)

    def close(self) -> None:
        if self.mode == AppMode.CALIBRATE:
            self._cancel_calibration()
        self._disconnect_guidance()
        self.closed = True
