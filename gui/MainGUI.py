"""GolfBot Main GUI — camera + 2D schematic with mode controls.

Single-window OpenCV GUI: live top-down camera feed (left) beside the 2D
schematic field view (right), with a status bar showing mode, pose, and FPS.
Guidance isolation testing (Stage 2) is available via the Guide Test button.
"""

from __future__ import annotations

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
from guidance.guidance import GuidanceController, GuidanceStatus
from localization.localization import RobotCalibrationCollector, RobotPoseEstimator
from localization.models import RobotPose
from path.pathfinding.models import HybridPose, PlannedBallTarget
from path.pathfinding.planner import RoutePlanningFacade
from config import AppConfig
from perception.vision.debug import DebugRenderer
from perception.vision.models import CalibrationState
from perception.vision.pipeline import VisionPipeline, VisionFrameResult


class AppMode(str, Enum):
    IDLE = "IDLE"
    MANUAL = "MANUAL"
    AUTO = "AUTO"
    GUIDANCE_TEST = "GUIDANCE_TEST"


WINDOW_NAME = "GolfBot Main"
CORNER_WINDOW_NAME = "Set Field Corners"
STATUS_BAR_HEIGHT = 130


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


def _draw_corner_overlay(frame: np.ndarray, state: CornerSelectionState) -> np.ndarray:
    """Draw point markers, polylines, loupe, and help text on the selector view."""
    overlay = frame.copy()
    h, w = overlay.shape[:2]

    # Loupe
    crop_sz = 40
    scale = 5
    padding = 12
    crop_w = min(crop_sz, w)
    crop_h = min(crop_sz, h)
    cx, cy = state.cursor
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
        self._route_planner = RoutePlanningFacade(
            field_config=self.config.field,
            robot_config=self.config.robot,
            planner_config=self.config.planner,
        )

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
        return self._panel_height() + STATUS_BAR_HEIGHT

    def _panel_height(self) -> int:
        return max(self._left_h, self._right_h)

    def buttons(self) -> list[GuiButton]:
        # Buttons sit on a row below the status text lines
        y = self._panel_height() + 90
        bw, bh = 95, 30
        gap = 8
        x0 = 20
        return [
            GuiButton("Set Corners", "set_corners", (x0, y, bw, bh)),
            GuiButton("Guide Test", "guidance_test", (x0 + (bw + gap), y, bw, bh)),
            GuiButton("Manual", "manual", (x0 + 2 * (bw + gap), y, bw, bh)),
            GuiButton("Auto", "auto", (x0 + 3 * (bw + gap), y, bw, bh)),
            GuiButton("Stop", "stop", (x0 + 4 * (bw + gap), y, bw, bh)),
            GuiButton("Quit", "quit", (x0 + 5 * (bw + gap), y, bw, bh)),
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

    def _handle_button(self, action: str) -> None:
        if action == "set_corners":
            self._open_corner_window()
        elif action == "guidance_test":
            if self.mode == AppMode.GUIDANCE_TEST:
                self._cycle_test_route()
            else:
                self._start_guidance_test()
        elif action == "manual":
            self._disconnect_guidance()
            self.mode = AppMode.MANUAL
            self.message = "Manual mode (view only)"
        elif action == "auto":
            self._start_brain()
        elif action == "stop":
            self._disconnect_guidance()
            self.mode = AppMode.IDLE
            self.message = "Stopped"
        elif action == "quit":
            self.closed = True

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
            # Feed the 4 corners into the pipeline's HomographyCalibrator
            calibrator = self.pipeline.preprocessor.homography_calibrator
            calibrator.set_manual_points(self._corner_state.points)
            self.message = "Field corners set — top-down warp active"
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
        if self._brain is not None:
            self._brain.reset()
        self._brain = None
        self._brain_state = None
        self._brain_route_points = None
        self._last_brain_time = None
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

        self._connecting = True
        self.message = "Connecting and planning route..."

        def connect_and_plan() -> None:
            try:
                commander = RobotCommander(
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

                plan = self._route_planner.plan_route(
                    captured_grid, targets, start_pose, geometry,
                )

                if not plan.points:
                    self.message = "Planner returned empty route"
                    commander.close()
                    return

                brain.load_route(plan)

                self._commander = commander
                self._guidance = guidance
                self._brain = brain
                self._brain_route_points = plan.points
                self._last_brain_time = None
                self._brain_state = None
                self.mode = AppMode.AUTO
                self.message = f"Brain running — {brain.step_count} steps"
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

        self._brain_state = self._brain.tick(self.robot_pose, dt_s)

    # ------------------------------------------------------------------
    # Per-frame processing
    # ------------------------------------------------------------------

    def _process_frame(self, raw_frame: np.ndarray) -> VisionFrameResult:
        return self.pipeline.process(raw_frame, params=self.params, use_aruco=True)

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
            left = self.renderer.make_topdown_placeholder("No top-down warp — click Set Corners")

        if left.shape[1] != self._left_w or left.shape[0] != self._left_h:
            left = cv2.resize(left, (self._left_w, self._left_h), interpolation=cv2.INTER_LINEAR)
        return left

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
            brain_line = f"Brain: {bs} | Step: {step_cur}/{step_tot} | {connected}"
            cv2.putText(canvas, brain_line, (20, y0 + 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 255, 220), 1, cv2.LINE_AA)
            cv2.putText(canvas, self.message, (20, y0 + 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 220, 180), 1, cv2.LINE_AA)
        else:
            cv2.putText(canvas, self.message, (20, y0 + 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 220, 180), 1, cv2.LINE_AA)

        cv2.putText(canvas,
                    "Keys: q/Esc quit | f set corners | g guidance test | a auto | s stop",
                    (20, y0 + 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1, cv2.LINE_AA)

        # Buttons drawn on their own row below text (y computed in buttons())
        for button in self.buttons():
            bx, by, bw, bh = button.rect
            is_active = (
                (button.action == "set_corners" and self._corner_window_open)
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

        if raw_frame is not None:
            result = self._process_frame(raw_frame)
            self._last_result = result
            self._estimate_pose(result)
            self._tick_guidance()
            self._tick_brain()
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
            else:
                self.closed = True
        elif key == ord("f"):
            self._handle_button("set_corners")
        elif key == ord("g"):
            self._handle_button("guidance_test")
        elif key == ord("r") and self._corner_window_open:
            self._corner_state.points.clear()
        elif key == ord("m"):
            self._handle_button("manual")
        elif key == ord("a"):
            self._handle_button("auto")
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

        self._disconnect_guidance()
        if self._corner_window_open:
            self._close_corner_window()
        if self.camera is not None:
            self.camera.release()
        cv2.destroyWindow(WINDOW_NAME)

    def close(self) -> None:
        self._disconnect_guidance()
        self.closed = True
