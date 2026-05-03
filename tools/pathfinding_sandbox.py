"""Interactive standalone route-planning sandbox.

This tool deliberately avoids the camera and vision pipeline.  It lets a user
place synthetic route-planning inputs directly on the schematic field while
reusing the existing OOP planning, occupancy-grid, and debug-rendering classes.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pathfinding.models import HybridPose, PlannedBallTarget, RoutePlan
from pathfinding.planner import RoutePlanningFacade
from robot.models import RobotPose
from vision.config import AppConfig
from vision.debug import SchematicRenderer
from vision.grid_mapping import OccupancyGridBuilder
from vision.models import RedZoneDetection, SmoothedBallCoordinate

WINDOW_NAME = "Pathfinding Sandbox"
BALL_LABELS = ("white", "orange")


@dataclass(frozen=True)
class SandboxBall:
    """Synthetic ball placed manually in bottom-left field centimeters."""

    track_id: int
    label: str
    x_cm: float
    y_cm: float


@dataclass
class SandboxState:
    """Mutable UI state owned only by this standalone sandbox."""

    config: AppConfig
    renderer: SchematicRenderer
    mode: str = "robot"
    robot_pose: RobotPose | None = None
    balls: list[SandboxBall] = field(default_factory=list)
    next_ball_label: str = "white"
    next_track_id: int = 1
    obstacle_contours: list[np.ndarray] = field(default_factory=list)
    drag_start_px: tuple[int, int] | None = None
    drag_current_px: tuple[int, int] | None = None
    is_dragging: bool = False
    status: str = "Mode robot: click or drag to place robot"
    scenario_revision: int = 0
    planned_revision: int = -1
    cached_grid: np.ndarray | None = None
    cached_route: RoutePlan | None = None

    @property
    def source_size(self) -> tuple[int, int]:
        return self.config.windows.schematic_size_px

    @property
    def frame_shape(self) -> tuple[int, int, int]:
        width_px, height_px = self.source_size
        return (height_px, width_px, 3)


def clamp_schematic_point(state: SandboxState, x_px: int, y_px: int) -> tuple[int, int]:
    """Clamp a mouse position to the schematic canvas."""
    return (
        int(np.clip(x_px, 0, state.config.windows.schematic_width_px - 1)),
        int(np.clip(y_px, 0, state.config.windows.schematic_height_px - 1)),
    )


def contour_from_rect(start_px: tuple[int, int], end_px: tuple[int, int]) -> np.ndarray | None:
    """Create an OpenCV contour from two schematic rectangle corners."""
    x0, y0 = start_px
    x1, y1 = end_px
    left, right = sorted((x0, x1))
    top, bottom = sorted((y0, y1))
    if right - left < 2 or bottom - top < 2:
        return None
    return np.array(
        [
            [[left, top]],
            [[right, top]],
            [[right, bottom]],
            [[left, bottom]],
        ],
        dtype=np.int32,
    )


def red_zone_from_contour(contour: np.ndarray) -> RedZoneDetection:
    """Wrap a manually drawn contour in the perception model expected downstream."""
    moments = cv2.moments(contour)
    if abs(moments["m00"]) > 1e-9:
        center = (int(round(moments["m10"] / moments["m00"])), int(round(moments["m01"] / moments["m00"])))
    else:
        x, y, w, h = cv2.boundingRect(contour)
        center = (x + w // 2, y + h // 2)

    return RedZoneDetection(
        contour=contour,
        corrected_contour=contour,
        bounding_box=cv2.boundingRect(contour),
        center=center,
        corrected_center=center,
        area=float(cv2.contourArea(contour)),
    )


def build_red_zones(state: SandboxState) -> list[RedZoneDetection]:
    """Convert sandbox obstacle contours into red-zone detections."""
    return [red_zone_from_contour(contour) for contour in state.obstacle_contours]


def balls_as_smoothed_coordinates(state: SandboxState) -> list[SmoothedBallCoordinate]:
    """Convert manually placed balls into the renderer's ball model."""
    coordinates: list[SmoothedBallCoordinate] = []
    for ball in state.balls:
        center_px = state.renderer.mapper.field_metric_cm_to_schematic((ball.x_cm, ball.y_cm))
        coordinates.append(
            SmoothedBallCoordinate(
                track_id=ball.track_id,
                label=ball.label,
                center_px=center_px,
                corrected_center_px=center_px,
                radius_px=8,
                cm_x=ball.x_cm,
                cm_y=ball.y_cm,
            )
        )
    return coordinates


def balls_as_planned_targets(state: SandboxState) -> list[PlannedBallTarget]:
    """Convert manually placed balls into route planner target models."""
    targets: list[PlannedBallTarget] = []
    for ball in state.balls:
        targets.append(
            PlannedBallTarget(
                track_id=ball.track_id,
                label=ball.label,
                x_cm=ball.x_cm,
                y_cm=ball.y_cm,
                node_cm=state.renderer.mapper.field_metric_cm_to_grid_node((ball.x_cm, ball.y_cm)),
            )
        )
    return targets


def make_robot_pose(state: SandboxState, origin_cm: tuple[float, float], heading_rad: float) -> RobotPose:
    """Build a full RobotPose, including tube coordinates, from a sandbox click."""
    geometry = state.renderer.robot_geometry_from_params(None)
    tube_x_cm = origin_cm[0] + math.cos(heading_rad) * geometry.tube_forward_cm
    tube_y_cm = origin_cm[1] + math.sin(heading_rad) * geometry.tube_forward_cm
    return RobotPose(
        x_cm=origin_cm[0],
        y_cm=origin_cm[1],
        heading_rad=heading_rad,
        tube_x_cm=float(np.clip(tube_x_cm, 0.0, state.config.field.width_cm)),
        tube_y_cm=float(np.clip(tube_y_cm, 0.0, state.config.field.height_cm)),
    )


def toggled_ball_label(label: str) -> str:
    """Return the next supported ball color label."""
    return "orange" if label == "white" else "white"


def nearest_ball_index(state: SandboxState, point_px: tuple[int, int], max_distance_px: float = 18.0) -> int | None:
    """Find the nearest manually placed ball to a schematic pixel."""
    best_index: int | None = None
    best_distance = max_distance_px
    for index, ball in enumerate(state.balls):
        ball_px = state.renderer.mapper.field_metric_cm_to_schematic((ball.x_cm, ball.y_cm))
        distance = math.hypot(ball_px[0] - point_px[0], ball_px[1] - point_px[1])
        if distance <= best_distance:
            best_distance = distance
            best_index = index
    return best_index


def toggle_ball_color(state: SandboxState, index: int) -> None:
    """Toggle one existing ball between white and orange."""
    ball = state.balls[index]
    next_label = toggled_ball_label(ball.label)
    state.balls[index] = SandboxBall(
        track_id=ball.track_id,
        label=next_label,
        x_cm=ball.x_cm,
        y_cm=ball.y_cm,
    )
    state.scenario_revision += 1
    state.status = f"Ball {ball.track_id} changed to {next_label}"


def reset_scenario(state: SandboxState) -> None:
    """Clear all manually placed route-planning inputs."""
    state.robot_pose = None
    state.balls.clear()
    state.obstacle_contours.clear()
    state.drag_start_px = None
    state.drag_current_px = None
    state.is_dragging = False
    state.next_track_id = 1
    state.scenario_revision += 1
    state.status = "Cleared scenario"


def handle_mouse(event: int, x_px: int, y_px: int, _flags: int, userdata: SandboxState) -> None:
    """OpenCV mouse callback implementing the sandbox interaction modes."""
    state = userdata
    point_px = clamp_schematic_point(state, x_px, y_px)

    if event == cv2.EVENT_RBUTTONUP:
        ball_index = nearest_ball_index(state, point_px)
        if ball_index is not None:
            toggle_ball_color(state, ball_index)
        else:
            state.status = "Right-click missed: no nearby ball"
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        state.drag_start_px = point_px
        state.drag_current_px = point_px
        state.is_dragging = True
        return

    if event == cv2.EVENT_MOUSEMOVE and state.is_dragging:
        state.drag_current_px = point_px
        return

    if event != cv2.EVENT_LBUTTONUP or state.drag_start_px is None:
        return

    state.drag_current_px = point_px
    start_px = state.drag_start_px
    end_px = point_px
    state.is_dragging = False

    if state.mode == "robot":
        origin_cm = state.renderer.mapper.schematic_to_field_metric_cm(start_px)
        drag_dx = end_px[0] - start_px[0]
        drag_dy = end_px[1] - start_px[1]
        if math.hypot(drag_dx, drag_dy) >= 5.0:
            end_cm = state.renderer.mapper.schematic_to_field_metric_cm(end_px)
            heading_rad = math.atan2(end_cm[1] - origin_cm[1], end_cm[0] - origin_cm[0])
        else:
            heading_rad = 0.0
        state.robot_pose = make_robot_pose(state, origin_cm, heading_rad)
        state.scenario_revision += 1
        state.status = f"Robot at {origin_cm[0]:.1f}, {origin_cm[1]:.1f} cm"
    elif state.mode == "target":
        ball_cm = state.renderer.mapper.schematic_to_field_metric_cm(end_px)
        state.balls.append(
            SandboxBall(
                track_id=state.next_track_id,
                label=state.next_ball_label,
                x_cm=ball_cm[0],
                y_cm=ball_cm[1],
            )
        )
        state.next_track_id += 1
        state.scenario_revision += 1
        state.status = f"Added {state.next_ball_label} ball at {ball_cm[0]:.1f}, {ball_cm[1]:.1f} cm"
    elif state.mode == "obstacle":
        contour = contour_from_rect(start_px, end_px)
        if contour is not None:
            state.obstacle_contours.append(contour)
            state.scenario_revision += 1
            state.status = f"Added obstacle {len(state.obstacle_contours)}"
        else:
            state.status = "Obstacle ignored: rectangle too small"

    state.drag_start_px = None
    state.drag_current_px = None


def draw_grid_overlay(image: np.ndarray, state: SandboxState, spacing_cm: int = 10) -> None:
    """Draw a light centimeter grid on top of the schematic image."""
    field = state.config.field
    mapper = state.renderer.mapper
    for x_cm in range(0, field.grid_width_cm + 1, spacing_cm):
        top = mapper.field_metric_cm_to_schematic((float(x_cm), field.height_cm))
        bottom = mapper.field_metric_cm_to_schematic((float(x_cm), 0.0))
        cv2.line(image, top, bottom, (65, 125, 65), 1, cv2.LINE_AA)
    for y_cm in range(0, field.grid_height_cm + 1, spacing_cm):
        left = mapper.field_metric_cm_to_schematic((0.0, float(y_cm)))
        right = mapper.field_metric_cm_to_schematic((field.width_cm, float(y_cm)))
        cv2.line(image, left, right, (65, 125, 65), 1, cv2.LINE_AA)


def draw_drag_preview(image: np.ndarray, state: SandboxState) -> None:
    """Draw the active drag gesture before it is committed."""
    if not state.is_dragging or state.drag_start_px is None or state.drag_current_px is None:
        return

    if state.mode == "robot":
        cv2.arrowedLine(image, state.drag_start_px, state.drag_current_px, (255, 255, 255), 2, cv2.LINE_AA, tipLength=0.25)
    elif state.mode == "obstacle":
        contour = contour_from_rect(state.drag_start_px, state.drag_current_px)
        if contour is not None:
            cv2.polylines(image, [contour], True, (0, 0, 255), 2, cv2.LINE_AA)
    elif state.mode == "target":
        fill_color, edge_color = ball_colors(state.next_ball_label)
        cv2.circle(image, state.drag_current_px, 8, fill_color, -1, cv2.LINE_AA)
        cv2.circle(image, state.drag_current_px, 8, edge_color, 1, cv2.LINE_AA)


def ball_colors(label: str) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Return BGR fill and edge colors matching the schematic renderer."""
    if label == "orange":
        return (0, 140, 255), (0, 80, 180)
    return (245, 245, 245), (120, 120, 120)


def draw_help_text(image: np.ndarray, state: SandboxState, route_plan: RoutePlan | None) -> None:
    """Draw compact sandbox controls and route status."""
    route_points = 0 if route_plan is None else len(route_plan.points)
    lines = [
        "r: robot   t: add ball   o: obstacle   1: white   2: orange   x/right-click: toggle ball   c: clear",
        (
            f"Mode: {state.mode}   Next ball: {state.next_ball_label}   Balls: {len(state.balls)}   "
            f"Obstacles: {len(state.obstacle_contours)}   Route points: {route_points}"
        ),
        state.status,
    ]

    y_px = image.shape[0] - 58
    for line in lines:
        cv2.putText(image, line, (16, y_px), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, line, (16, y_px), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y_px += 22


def compute_route(
    state: SandboxState,
    occupancy_builder: OccupancyGridBuilder,
    route_planner: RoutePlanningFacade,
) -> tuple[np.ndarray, RoutePlan | None]:
    """Build occupancy and route from the current manual scenario."""
    if state.planned_revision == state.scenario_revision and state.cached_grid is not None:
        return state.cached_grid, state.cached_route

    red_zones = build_red_zones(state)
    raw_grid = occupancy_builder.build(state.frame_shape, red_zones, dilate_for_legacy=False)

    if state.robot_pose is None or not state.balls:
        state.cached_grid = raw_grid
        state.cached_route = None
        state.planned_revision = state.scenario_revision
        return state.cached_grid, state.cached_route

    start_pose = HybridPose(
        x_cm=state.robot_pose.x_cm,
        y_cm=state.robot_pose.y_cm,
        theta_rad=state.robot_pose.heading_rad,
    )
    route_plan = route_planner.plan_route(
        raw_grid,
        balls_as_planned_targets(state),
        start_pose,
        state.renderer.robot_geometry_from_params(None),
    )
    state.cached_grid = raw_grid
    state.cached_route = route_plan
    state.planned_revision = state.scenario_revision
    return state.cached_grid, state.cached_route


def render_sandbox(state: SandboxState, route_plan: RoutePlan | None) -> np.ndarray:
    """Render the full sandbox view using the existing schematic renderer."""
    red_zones = build_red_zones(state)
    route_points = None if route_plan is None else route_plan.points
    pickup_poses = None if route_plan is None else route_plan.pickup_poses
    camera_center = (
        state.config.windows.schematic_width_px * 0.5,
        state.config.windows.schematic_height_px * 0.5,
    )

    image = state.renderer.draw_schematic(
        frame_shape=state.frame_shape,
        red_zones=red_zones,
        smoothed_ball_coordinates=balls_as_smoothed_coordinates(state),
        camera_center_pixels=camera_center,
        route_points_cm=route_points,
        route_pickup_poses_cm=pickup_poses,
        robot_pose=state.robot_pose,
        params=None,
        num_intermediate_snapshots=state.config.planner.num_intermediate_snapshots,
        route_heading_marker_interval=state.config.planner.route_heading_marker_interval,
    )
    draw_grid_overlay(image, state)
    draw_drag_preview(image, state)
    draw_help_text(image, state, route_plan)
    return image


def set_mode(state: SandboxState, mode: str) -> None:
    """Switch interaction mode and update the visible status string."""
    state.mode = mode
    state.drag_start_px = None
    state.drag_current_px = None
    state.is_dragging = False
    if mode == "robot":
        state.status = "Mode robot: click or drag to place heading"
    elif mode == "target":
        state.status = f"Mode target: click to add a {state.next_ball_label} ball"
    elif mode == "obstacle":
        state.status = "Mode obstacle: drag rectangles to add red zones"


def set_next_ball_label(state: SandboxState, label: str) -> None:
    """Set the color label used by subsequently placed balls."""
    if label not in BALL_LABELS:
        return
    state.next_ball_label = label
    state.status = f"Next ball color: {label}"


def toggle_last_ball_color(state: SandboxState) -> None:
    """Toggle the most recently placed ball color."""
    if not state.balls:
        state.status = "No balls to recolor"
        return
    toggle_ball_color(state, len(state.balls) - 1)


def main() -> int:
    """Run the OpenCV sandbox loop."""
    config = AppConfig.from_repo_root(REPO_ROOT)
    renderer = SchematicRenderer(config.field, config.windows, config.robot)
    occupancy_builder = OccupancyGridBuilder(config.field, config.robot, renderer.mapper)
    route_planner = RoutePlanningFacade(config.field, config.robot, config.planner)
    state = SandboxState(config=config, renderer=renderer)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, config.windows.schematic_width_px, config.windows.schematic_height_px)
    cv2.setMouseCallback(WINDOW_NAME, handle_mouse, state)

    while True:
        _raw_grid, route_plan = compute_route(state, occupancy_builder, route_planner)
        cv2.imshow(WINDOW_NAME, render_sandbox(state, route_plan))

        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord("r"):
            set_mode(state, "robot")
        elif key == ord("t"):
            set_mode(state, "target")
        elif key == ord("o"):
            set_mode(state, "obstacle")
        elif key == ord("1"):
            set_next_ball_label(state, "white")
        elif key == ord("2"):
            set_next_ball_label(state, "orange")
        elif key == ord("x"):
            toggle_last_ball_color(state)
        elif key == ord("c"):
            reset_scenario(state)

    cv2.destroyWindow(WINDOW_NAME)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
