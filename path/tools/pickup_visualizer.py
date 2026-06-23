"""Pickup-point geometry visualizer — fit-based pathing edition.

Computes and renders the pickup-ring geometric model with obstacle-clearance
classification for every ball on the field.  Two input modes:

    --mode=test     Deterministic test scenario with random ball placements
                    (no camera required).
    --mode=camera   Live camera capture using the existing perception pipeline.

The rendered image is saved to ``path/tools/output/pickup_geometry.png`` and
displayed in an interactive OpenCV window.
"""

from __future__ import annotations

import argparse
import math
import sys
from enum import Enum
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import AppConfig, FieldConfig
from path.models import HybridPose, PlannedBallTarget, RoutePlan, WaypointKind
from path.pickup_geometry import (
    PickupCategory,
    PickupGeometryResult,
    compute_pickup_geometry,
    pickup_reach_cm,
)
from path.planner import CompileStrategy, compile_route
from path.route_stats import RouteStats, compute_route_stats
from path.route_strategy import (
    IntersectionPriorityStrategy,
    NearestNeighborStrategy,
    RoutePlannerInput,
    RouteStrategyResult,
    SweepStrategy,
    TwoOptStrategy,
)
from path.tools.pathfinding_sandbox import (
    RANDOM_BALL_COUNT,
    SandboxBall,
    SandboxState,
    balls_as_smoothed_coordinates,
    build_red_zones,
    centered_cross_obstacle_contours,
    is_inside_center_cross_clearance,
)
from perception.vision.debug import SchematicRenderer
from perception.vision.geometry import CoordinateMapper
from perception.vision.grid_mapping import OccupancyGridBuilder

WINDOW_NAME = "Pickup Geometry Visualizer"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_FILE = OUTPUT_DIR / "pickup_geometry.png"


class OverlayMode(Enum):
    """Which background overlay to render behind pickup candidates."""

    HEATMAP = "heatmap"      # continuous distance field (JET colormap)
    COLLISION = "collision"  # binary blocked grid (robot body clearance)
    NONE = "none"            # no overlay


# Category colors (BGR)
CATEGORY_COLORS = {
    PickupCategory.SAFE: (0, 220, 0),        # green
    PickupCategory.IN_BETWEEN: (0, 220, 220), # yellow
    PickupCategory.CONSTRAINED: (0, 0, 220),  # red
}
INVALID_COLOR = (30, 30, 30)  # near-black for invalid headings


class StrategyOption:
    """One selectable visualizer routing strategy."""

    def __init__(self, key: str, label: str, strategy_cls) -> None:
        self.key = key
        self.label = label
        self.strategy_cls = strategy_cls

    def create(self):
        return self.strategy_cls()


STRATEGY_OPTIONS = (
    StrategyOption("nearest-neighbor", "Nearest-neighbor (fit-based)", NearestNeighborStrategy),
    StrategyOption("2-opt", "2-opt (distance)", TwoOptStrategy),
    StrategyOption("sweep", "Sweep (turns)", SweepStrategy),
    StrategyOption("intersection", "Intersection priority", IntersectionPriorityStrategy),
)


# ---------------------------------------------------------------------------
# Test scenario generator
# ---------------------------------------------------------------------------

def _mixed_ball_point(
    rng: np.random.Generator,
    width: float,
    height: float,
) -> tuple[float, float]:
    """Sample a ball position from a mixed distribution.

    Distribution mix (approximate):
      ~30 % near field edges  (10-25 cm from wall)
      ~30 % near the cross    (within ~20 cm of the cross perimeter)
      ~40 % uniform interior  (margin-padded)

    This gives more realistic test coverage than pure edge-biased
    placement: balls cluster near the cross (where routing is hard)
    while still appearing throughout the field.
    """
    margin = 10.0
    r = float(rng.uniform())

    if r < 0.30:
        # --- Edge-biased: 10-25 cm from a random wall ---
        perimeter = 2 * (width + height)
        p = float(rng.uniform(0, perimeter))
        offset = float(rng.uniform(10.0, 25.0))
        if p < width:
            return float(rng.uniform(margin, width - margin)), offset
        p -= width
        if p < height:
            return width - offset, float(rng.uniform(margin, height - margin))
        p -= height
        if p < width:
            return float(rng.uniform(margin, width - margin)), height - offset
        return offset, float(rng.uniform(margin, height - margin))

    if r < 0.60:
        # --- Cross-biased: near the cross perimeter ---
        cx, cy = width * 0.5, height * 0.5
        # Random angle, distance 12-25 cm from cross center.
        angle = float(rng.uniform(0, 2 * math.pi))
        dist = float(rng.uniform(12.0, 25.0))
        x = cx + dist * math.cos(angle)
        y = cy + dist * math.sin(angle)
        # Clamp inside field margins.
        x = max(margin, min(width - margin, x))
        y = max(margin, min(height - margin, y))
        return x, y

    # --- Uniform interior ---
    return (
        float(rng.uniform(margin, width - margin)),
        float(rng.uniform(margin, height - margin)),
    )


# Minimum distance (cm) between any two balls in the test scenario.
_MIN_BALL_SPACING_CM = 13.0


def generate_test_scenario(
    state: SandboxState,
    seed: int = 42,
) -> None:
    """Populate the sandbox with randomly placed balls and a center cross.

    Uses a mixed placement distribution: balls appear near the field
    edges, near the cross obstacle, and uniformly across the interior.
    """
    state.rng = np.random.default_rng(seed)
    state.balls.clear()
    state.obstacle_contours = centered_cross_obstacle_contours(state)
    state.next_track_id = 1

    field = state.config.field
    positions: list[tuple[float, float]] = []
    max_attempts = 2000
    for _ in range(max_attempts):
        if len(positions) >= RANDOM_BALL_COUNT:
            break
        point = _mixed_ball_point(state.rng, field.width_cm, field.height_cm)
        if is_inside_center_cross_clearance(state, point, 0.0):
            continue
        if any(
            math.hypot(point[0] - ex[0], point[1] - ex[1]) < _MIN_BALL_SPACING_CM
            for ex in positions
        ):
            continue
        positions.append(point)

    orange_index = int(state.rng.integers(0, len(positions)))

    for index, (x_cm, y_cm) in enumerate(positions):
        label = "orange" if index == orange_index else "white"
        state.balls.append(
            SandboxBall(
                track_id=state.next_track_id,
                label=label,
                x_cm=x_cm,
                y_cm=y_cm,
            )
        )
        state.next_track_id += 1


def balls_as_planned_targets(state: SandboxState) -> list[PlannedBallTarget]:
    """Convert sandbox balls to the planner's target model."""
    targets: list[PlannedBallTarget] = []
    for ball in state.balls:
        targets.append(
            PlannedBallTarget(
                track_id=ball.track_id,
                label=ball.label,
                x_cm=ball.x_cm,
                y_cm=ball.y_cm,
                node_cm=state.renderer.mapper.field_metric_cm_to_grid_node(
                    (ball.x_cm, ball.y_cm)
                ),
            )
        )
    return targets


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _cm_to_px(cm: float, field_config: FieldConfig, schematic_width_px: int) -> int:
    """Convert a distance in centimetres to schematic pixels."""
    return max(1, int(round(cm * (schematic_width_px - 1) / max(1.0, field_config.width_cm))))


def draw_pickup_geometry(
    image: np.ndarray,
    result: PickupGeometryResult,
    mapper: CoordinateMapper,
    field_config: FieldConfig,
    overlay_mode: OverlayMode = OverlayMode.HEATMAP,
    half_width_cm: float = 0.0,
) -> np.ndarray:
    """Overlay pickup geometry onto an existing schematic image.

    Draws distance-field heatmap or collision map, per-ball rings, and
    color-coded candidates (green=SAFE, yellow=IN_BETWEEN, red=CONSTRAINED).

    Parameters
    ----------
    overlay_mode:
        HEATMAP  — continuous distance field (JET colormap).
        COLLISION — binary blocked grid (red = robot body doesn't fit).
        NONE     — no background overlay.
    half_width_cm:
        Robot half-width for the collision map threshold.  Only used when
        *overlay_mode* is COLLISION.
    """
    h, w = image.shape[:2]

    df = result.distance_field
    if df.size > 0 and overlay_mode == OverlayMode.HEATMAP:
        # --- Distance field heatmap overlay ---
        max_d = max(result.tank_turn_radius_cm * 1.5, 1.0)
        normalized = np.clip(df / max_d, 0.0, 1.0)
        resized = cv2.resize(normalized.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        heatmap_u8 = (resized * 255).astype(np.uint8)
        heatmap_bgr = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
        cv2.addWeighted(heatmap_bgr, 0.15, image, 0.85, 0, image)
    elif df.size > 0 and overlay_mode == OverlayMode.COLLISION:
        # --- Binary collision map overlay ---
        blocked = (df < half_width_cm).astype(np.uint8)
        resized = cv2.resize(blocked, (w, h), interpolation=cv2.INTER_NEAREST)
        overlay = np.zeros_like(image)
        overlay[resized > 0] = (0, 0, 200)   # red = blocked
        overlay[resized == 0] = (80, 50, 0)   # dark blue-gray = free
        cv2.addWeighted(overlay, 0.35, image, 0.65, 0, image)

    # --- Per-ball rings and candidates ---
    for ball_cands in result.balls:
        ball = ball_cands.ball
        ball_px = mapper.field_metric_cm_to_schematic((ball.x_cm, ball.y_cm))
        ring_radius_px = _cm_to_px(result.ring_radius_cm, field_config, w)

        # Ring circle
        ring_color = (255, 200, 0) if ball_cands.reachable else (0, 0, 200)
        cv2.circle(image, ball_px, ring_radius_px, ring_color, 1, cv2.LINE_AA)

        # Invalid headings — black dots
        for inv in ball_cands.invalid_headings:
            pt = mapper.field_metric_cm_to_schematic((inv.x_cm, inv.y_cm))
            cv2.circle(image, pt, 2, INVALID_COLOR, -1, cv2.LINE_AA)

        # Candidates color-coded by category
        for cand in ball_cands.candidates:
            pt = mapper.field_metric_cm_to_schematic((cand.x_cm, cand.y_cm))
            color = CATEGORY_COLORS[cand.category]
            cv2.circle(image, pt, 3, color, -1, cv2.LINE_AA)

        # Unreachable: red X over ball
        if not ball_cands.reachable:
            size = 8
            cv2.line(image,
                     (ball_px[0] - size, ball_px[1] - size),
                     (ball_px[0] + size, ball_px[1] + size),
                     (0, 0, 255), 2, cv2.LINE_AA)
            cv2.line(image,
                     (ball_px[0] - size, ball_px[1] + size),
                     (ball_px[0] + size, ball_px[1] - size),
                     (0, 0, 255), 2, cv2.LINE_AA)

    return image


def draw_route_plan(
    image: np.ndarray,
    strategy_result: RouteStrategyResult,
    geometry_result: PickupGeometryResult,
    mapper: CoordinateMapper,
    field_config: FieldConfig,
    start_pose: HybridPose | None = None,
    route_plan: RoutePlan | None = None,
) -> np.ndarray:
    """Overlay route plan visualization onto an existing schematic image.

    When *route_plan* is provided, the polyline follows the compiled Layer 3
    waypoints (which include A* detour segments around obstacles).  Otherwise
    falls back to straight lines between Layer 2 strategy stops.
    """
    h, w = image.shape[:2]
    stops = strategy_result.stops

    def to_px(x: float, y: float) -> tuple[int, int]:
        return mapper.field_metric_cm_to_schematic((x, y))

    # --- Intermediate nodes (small squares) connected to pickup points ---
    for stop in stops:
        if stop.intermediate_node is not None:
            inter = stop.intermediate_node
            cand = stop.candidate
            inter_px = to_px(inter.x_cm, inter.y_cm)
            cand_px = to_px(cand.x_cm, cand.y_cm)
            # Dashed line from intermediate to pickup
            _draw_dotted_line(image, inter_px, cand_px, (180, 180, 180), 1)
            # Small square at intermediate
            sz = 4
            cv2.rectangle(image,
                          (inter_px[0] - sz, inter_px[1] - sz),
                          (inter_px[0] + sz, inter_px[1] + sz),
                          (255, 180, 0), -1, cv2.LINE_AA)
            cv2.rectangle(image,
                          (inter_px[0] - sz, inter_px[1] - sz),
                          (inter_px[0] + sz, inter_px[1] + sz),
                          (0, 0, 0), 1, cv2.LINE_AA)

    # --- Route polyline with arrowheads ---
    if route_plan is not None:
        # Use compiled Layer 3 waypoints (collision-free).
        route_waypoints: list[tuple[float, float]] = []
        if start_pose is not None:
            route_waypoints.append((start_pose.x_cm, start_pose.y_cm))
        for wp in route_plan.waypoints:
            route_waypoints.append((wp.x_cm, wp.y_cm))
    else:
        # Fallback: straight lines between Layer 2 strategy stops.
        route_waypoints = []
        if start_pose is not None:
            route_waypoints.append((start_pose.x_cm, start_pose.y_cm))
        for stop in stops:
            if stop.intermediate_node is not None:
                route_waypoints.append((stop.intermediate_node.x_cm, stop.intermediate_node.y_cm))
            route_waypoints.append((stop.candidate.x_cm, stop.candidate.y_cm))
            if stop.intermediate_node is not None:
                route_waypoints.append((stop.intermediate_node.x_cm, stop.intermediate_node.y_cm))
        if strategy_result.unload_position is not None:
            route_waypoints.append(strategy_result.unload_position)

    route_px = [to_px(x, y) for x, y in route_waypoints]
    for i in range(len(route_px) - 1):
        cv2.line(image, route_px[i], route_px[i + 1], (255, 220, 0), 2, cv2.LINE_AA)
        # Arrowhead at midpoint
        mid_x = (route_px[i][0] + route_px[i + 1][0]) // 2
        mid_y = (route_px[i][1] + route_px[i + 1][1]) // 2
        dx = route_px[i + 1][0] - route_px[i][0]
        dy = route_px[i + 1][1] - route_px[i][1]
        length = math.hypot(dx, dy)
        if length > 1e-3:
            ux, uy = dx / length, dy / length
            arrow_size = 6
            tip = (mid_x + int(ux * arrow_size), mid_y + int(uy * arrow_size))
            left = (mid_x - int(ux * arrow_size - uy * arrow_size * 0.5),
                    mid_y - int(uy * arrow_size + ux * arrow_size * 0.5))
            right_pt = (mid_x - int(ux * arrow_size + uy * arrow_size * 0.5),
                        mid_y - int(uy * arrow_size - ux * arrow_size * 0.5))
            cv2.fillPoly(image, [np.array([tip, left, right_pt], dtype=np.int32)], (255, 220, 0))

    # --- Stop markers with order numbers ---
    for visit_order, stop in enumerate(stops):
        px = to_px(stop.candidate.x_cm, stop.candidate.y_cm)
        color = CATEGORY_COLORS[stop.candidate.category]
        cv2.circle(image, px, 8, color, -1, cv2.LINE_AA)
        cv2.circle(image, px, 8, (0, 0, 0), 1, cv2.LINE_AA)
        label = str(visit_order + 1)
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        tx = px[0] - text_size[0] // 2
        ty = px[1] + text_size[1] // 2
        cv2.putText(image, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

    # Unload marker
    if strategy_result.unload_position is not None:
        ux, uy = strategy_result.unload_position
        up = to_px(ux, uy)
        tri = np.array([
            [up[0], up[1] - 7],
            [up[0] + 6, up[1] + 5],
            [up[0] - 6, up[1] + 5],
        ], dtype=np.int32)
        cv2.fillPoly(image, [tri], (255, 0, 255))
        cv2.polylines(image, [tri], True, (0, 0, 0), 1, cv2.LINE_AA)

    return image


def draw_legend(
    image: np.ndarray,
    geometry_result: PickupGeometryResult,
    strategy_result: RouteStrategyResult,
    strategy_label: str = "",
    compile_label: str = "",
) -> None:
    """Draw a compact legend at the bottom of the image."""
    h, w = image.shape[:2]
    stops = strategy_result.stops
    result = geometry_result

    total = len(result.balls)
    reachable = sum(1 for b in result.balls if b.reachable)
    intermediates = sum(1 for s in stops if s.intermediate_node is not None)

    # Single compact line
    legend = (
        f"{len(stops)} stops  {intermediates} intermediate  "
        f"{total - reachable} unreachable  |  "
        f"safe>{result.safe_radius_cm:.0f}cm  constrained<{result.constrained_radius_cm:.0f}cm"
    )

    y_px = h - 14
    cv2.putText(image, legend, (10, y_px), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, legend, (10, y_px), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (220, 220, 220), 1, cv2.LINE_AA)

    # Strategy name (top-left corner)
    if strategy_label:
        cv2.putText(image, strategy_label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, strategy_label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 220, 255), 1, cv2.LINE_AA)

    # Compile strategy (below strategy name)
    if compile_label:
        cv2.putText(image, compile_label, (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, compile_label, (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (180, 180, 255), 1, cv2.LINE_AA)


def draw_start_marker(
    image: np.ndarray,
    start_pose: HybridPose,
    mapper: CoordinateMapper,
) -> None:
    """Draw a small arrow at the robot's starting position."""
    px = mapper.field_metric_cm_to_schematic((start_pose.x_cm, start_pose.y_cm))
    arrow_len = 14
    tip = (
        px[0] + int(math.cos(start_pose.theta_rad) * arrow_len),
        px[1] - int(math.sin(start_pose.theta_rad) * arrow_len),
    )
    cv2.circle(image, px, 5, (255, 90, 30), -1, cv2.LINE_AA)
    cv2.arrowedLine(image, px, tip, (255, 90, 30), 2, cv2.LINE_AA, tipLength=0.4)


def _draw_dotted_line(
    image: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int,
    gap: int = 6,
) -> None:
    """Draw a dotted line between two pixel points."""
    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]
    length = math.hypot(dx, dy)
    if length < 1:
        return
    steps = int(length / gap)
    for i in range(0, steps, 2):
        t0 = i * gap / length
        t1 = min((i + 1) * gap / length, 1.0)
        p0 = (int(pt1[0] + dx * t0), int(pt1[1] + dy * t0))
        p1 = (int(pt1[0] + dx * t1), int(pt1[1] + dy * t1))
        cv2.line(image, p0, p1, color, thickness, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Base schematic rendering (reuses existing renderer)
# ---------------------------------------------------------------------------

def render_base_schematic(
    state: SandboxState,
) -> np.ndarray:
    """Draw the field, obstacles, and balls using the existing schematic renderer.

    Labels are suppressed — the visualizer draws its own compact legend.
    """
    red_zones = build_red_zones(state)
    camera_center = (
        state.config.windows.schematic_width_px * 0.5,
        state.config.windows.schematic_height_px * 0.5,
    )
    return state.renderer.draw_schematic(
        frame_shape=state.frame_shape,
        red_zones=red_zones,
        smoothed_ball_coordinates=balls_as_smoothed_coordinates(state),
        camera_center_pixels=camera_center,
        robot_pose=None,
        params=None,
        show_labels=False,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _compute_and_render(
    state: SandboxState,
    occupancy_builder: OccupancyGridBuilder,
    geometry,
    seed: int,
    strategy_option: StrategyOption,
    compile_strategy: CompileStrategy = CompileStrategy.SINGLE_NODE,
    overlay_mode: OverlayMode = OverlayMode.HEATMAP,
    quiet: bool = False,
) -> tuple[np.ndarray, RoutePlan]:
    """Generate a scenario, compute geometry + route, render, and print summary.

    Returns ``(image, route_plan)``.  When *quiet* is True, suppresses
    per-ball detail output (used in benchmark mode).
    """
    config = state.config

    generate_test_scenario(state, seed=seed)

    # Build occupancy grid from obstacles
    red_zones = build_red_zones(state)
    raw_grid = occupancy_builder.build(state.frame_shape, red_zones, dilate_for_legacy=False)

    # Compute pickup geometry
    targets = balls_as_planned_targets(state)
    result = compute_pickup_geometry(
        field_width_cm=config.field.width_cm,
        field_height_cm=config.field.height_cm,
        obstacle_grid=raw_grid,
        balls=targets,
        geometry=geometry,
    )

    # Compute route plan.
    # Robot starts opposite the delivery node: right side of the field,
    # centered vertically, facing left (toward the unload wall).
    unload_reach = geometry.rear_cm + geometry.unload_extension_cm
    unload_pos = (unload_reach + 2.0, config.field.height_cm * 0.5)
    start = HybridPose(
        x_cm=config.field.width_cm - geometry.rear_cm - 2.0,
        y_cm=config.field.height_cm * 0.5,
        theta_rad=math.pi,
    )

    route_input = RoutePlannerInput(
        geometry_result=result,
        start_pose=start,
        field_width_cm=config.field.width_cm,
        field_height_cm=config.field.height_cm,
        unload_position=unload_pos,
    )
    strategy = strategy_option.create()
    strategy_result = strategy.plan(route_input)

    # Layer 3: compile route (collision-free waypoints via A*).
    route_plan = compile_route(
        strategy_result,
        distance_field=result.distance_field,
        start_pose=start,
        half_width_cm=geometry.width_cm * 0.5,
        field_height_cm=config.field.height_cm,
        strategy=compile_strategy,
        balls=targets,
    )

    # Render
    image = render_base_schematic(state)
    draw_pickup_geometry(
        image, result, state.renderer.mapper, config.field,
        overlay_mode=overlay_mode,
        half_width_cm=geometry.width_cm * 0.5,
    )
    draw_route_plan(image, strategy_result, result, state.renderer.mapper, config.field, start,
                    route_plan=route_plan)
    draw_start_marker(image, start, state.renderer.mapper)
    _COMPILE_LABELS = {
        CompileStrategy.SINGLE_NODE: "compile: single_node",
        CompileStrategy.MINIMAL: "compile: minimal",
        CompileStrategy.FULL: "compile: full",
    }
    draw_legend(image, result, strategy_result,
                strategy_label=strategy_option.label,
                compile_label=_COMPILE_LABELS.get(compile_strategy, compile_strategy.value))

    # Save output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUTPUT_FILE), image)

    # Route stats
    stats = compute_route_stats(route_plan)

    # Print summary
    if not quiet:
        R = pickup_reach_cm(geometry)
        print(f"\n--- seed={seed} strategy={strategy_option.key} compile={compile_strategy.value} ---")
        print(f"R = {R:.1f} cm  tank = {result.tank_turn_radius_cm:.1f} cm  "
              f"sweep = {result.tube_sweep_radius_cm:.1f} cm")
        print(f"Field: {config.field.width_cm} x {config.field.height_cm} cm")
        print(f"Balls: {len(result.balls)}")
        for i, bc in enumerate(result.balls):
            ball = bc.ball
            if bc.reachable:
                safe = sum(1 for c in bc.candidates if c.category == PickupCategory.SAFE)
                between = sum(1 for c in bc.candidates if c.category == PickupCategory.IN_BETWEEN)
                constrained = sum(1 for c in bc.candidates if c.category == PickupCategory.CONSTRAINED)
                status = f"{len(bc.candidates)} candidates ({safe}S/{between}B/{constrained}C)"
            else:
                status = "UNREACHABLE"
            print(f"  #{ball.track_id} ({ball.label}) at ({ball.x_cm:.1f}, {ball.y_cm:.1f}): {status}")

        # Route summary
        intermediates = sum(1 for s in strategy_result.stops if s.intermediate_node is not None)
        print(f"Route: {len(strategy_result.stops)} stops, {intermediates} intermediates, "
              f"{len(strategy_result.unreachable_balls)} unreachable")
        for i, stop in enumerate(strategy_result.stops):
            ball = result.balls[stop.ball_index].ball
            cat = stop.candidate.category.value
            inter_str = ""
            if stop.intermediate_node is not None:
                inter = stop.intermediate_node
                inter_str = f" via ({inter.x_cm:.1f}, {inter.y_cm:.1f})"
            print(f"  Stop {i + 1}: #{ball.track_id} at ({stop.candidate.x_cm:.1f}, "
                  f"{stop.candidate.y_cm:.1f}) [{cat}]{inter_str}")

        # Compiled route summary
        nav_count = sum(1 for w in route_plan.waypoints if w.kind == WaypointKind.NAVIGATE)
        print(f"Compiled: {len(route_plan.waypoints)} waypoints ({nav_count} navigate)")

    print(f"Stats: distance={stats.total_distance_cm}cm  turns={stats.turn_count}  "
          f"max_turn={stats.max_turn_deg}deg  waypoints={stats.waypoint_count} "
          f"({stats.navigate_count} nav)")

    return image, route_plan


def _run_benchmark(
    n_seeds: int,
    state: SandboxState,
    occupancy_builder: OccupancyGridBuilder,
    geometry,
    compile_strategy: CompileStrategy,
    start_seed: int = 42,
) -> None:
    """Run *n_seeds* scenarios across all strategies and print a comparison table."""
    # {strategy_key: [RouteStats, ...]}
    all_stats: dict[str, list[RouteStats]] = {opt.key: [] for opt in STRATEGY_OPTIONS}

    for seed in range(start_seed, start_seed + n_seeds):
        for option in STRATEGY_OPTIONS:
            _, route_plan = _compute_and_render(
                state, occupancy_builder, geometry, seed, option,
                compile_strategy=compile_strategy, quiet=True,
            )
            stats = compute_route_stats(route_plan)
            all_stats[option.key].append(stats)

    # Print comparison table.
    print("\n" + "=" * 80)
    print(f"BENCHMARK: {n_seeds} seeds (seed {start_seed}..{start_seed + n_seeds - 1})")
    print("=" * 80)
    header = f"{'Strategy':<20} {'Dist(cm)':>10} {'Turns':>8} {'MaxTurn':>10} {'WPs':>6} {'Nav':>6}"
    print(header)
    print("-" * len(header))

    for option in STRATEGY_OPTIONS:
        stats_list = all_stats[option.key]
        n = len(stats_list)
        avg_dist = sum(s.total_distance_cm for s in stats_list) / n
        avg_turns = sum(s.turn_count for s in stats_list) / n
        avg_max_turn = sum(s.max_turn_deg for s in stats_list) / n
        avg_wps = sum(s.waypoint_count for s in stats_list) / n
        avg_nav = sum(s.navigate_count for s in stats_list) / n
        print(f"{option.key:<20} {avg_dist:>10.1f} {avg_turns:>8.1f} "
              f"{avg_max_turn:>10.1f} {avg_wps:>6.1f} {avg_nav:>6.1f}")

    print("=" * 80)


def main() -> int:
    """Run the pickup geometry visualizer."""
    parser = argparse.ArgumentParser(description="Pickup-point geometry visualizer")
    parser.add_argument(
        "--mode",
        choices=["test", "camera"],
        default="test",
        help="Input mode: 'test' for synthetic scenario, 'camera' for live capture",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for test scenario")
    parser.add_argument(
        "--strategy",
        choices=[option.key for option in STRATEGY_OPTIONS],
        default=STRATEGY_OPTIONS[0].key,
        help="Initial route strategy shown in the visualizer",
    )
    parser.add_argument(
        "--compile",
        choices=["single_node", "full", "minimal"],
        default="single_node",
        help="Initial compile strategy: 'single_node' (one detour node), 'full' (Douglas-Peucker), or 'minimal' (greedy visibility)",
    )
    parser.add_argument(
        "--benchmark",
        type=int,
        default=0,
        metavar="N",
        help="Run N seeds across all strategies and print comparison table (no GUI)",
    )
    args = parser.parse_args()

    if args.mode == "camera":
        print("Camera mode not yet implemented -- falling back to test mode.")

    config = AppConfig.from_repo_root(REPO_ROOT)
    renderer = SchematicRenderer(config.field, config.windows, config.robot)
    occupancy_builder = OccupancyGridBuilder(config.field, config.robot, renderer.mapper)

    # Load geometry from the calibration file (single source of truth),
    # falling back to config defaults if the file is absent or incomplete.
    geo_params: dict[str, object] = {}
    cal_path = config.paths.robot_calibration_file
    if cal_path is not None and cal_path.exists():
        import json as _json
        with cal_path.open() as _f:
            _cal = _json.load(_f)
        from localization.localization import RobotCalibrationCollector
        RobotCalibrationCollector.apply_geometry_to_params(
            geo_params, _cal.get("geometry"),
        )
    geometry = renderer.robot_geometry_from_params(geo_params or None)
    state = SandboxState(config=config, renderer=renderer)

    compile_strategy = CompileStrategy(args.compile)

    # --- Benchmark mode (no GUI) ---
    if args.benchmark > 0:
        _run_benchmark(
            args.benchmark, state, occupancy_builder, geometry,
            compile_strategy, start_seed=args.seed,
        )
        return 0

    seed = args.seed
    strategy_index = next(
        index for index, option in enumerate(STRATEGY_OPTIONS)
        if option.key == args.strategy
    )
    image: np.ndarray | None = None
    overlay_mode = OverlayMode.HEATMAP
    _OVERLAY_CYCLE = [OverlayMode.HEATMAP, OverlayMode.COLLISION, OverlayMode.NONE]

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, config.windows.schematic_width_px, config.windows.schematic_height_px)

    HELP_TEXT = (
        "Press 'r' to randomize, 's' to cycle strategy, "
        "'c' to toggle compile strategy, 'v' to cycle overlay "
        "(heatmap/collision/none), Esc/q to quit."
    )

    def render_current() -> None:
        nonlocal image
        image, _ = _compute_and_render(
            state,
            occupancy_builder,
            geometry,
            seed,
            STRATEGY_OPTIONS[strategy_index],
            compile_strategy=compile_strategy,
            overlay_mode=overlay_mode,
        )
        cv2.imshow(WINDOW_NAME, image)

    render_current()
    print(f"\n{HELP_TEXT}")

    while True:
        key = cv2.waitKey(50) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord("r"):
            seed += 1
            render_current()
            print(f"\n{HELP_TEXT}")
        if key == ord("s"):
            strategy_index = (strategy_index + 1) % len(STRATEGY_OPTIONS)
            render_current()
            print(f"\n{HELP_TEXT}")
        if key == ord("c"):
            _compile_cycle = [CompileStrategy.SINGLE_NODE, CompileStrategy.MINIMAL, CompileStrategy.FULL]
            _ci = _compile_cycle.index(compile_strategy) if compile_strategy in _compile_cycle else -1
            compile_strategy = _compile_cycle[(_ci + 1) % len(_compile_cycle)]
            render_current()
            print(f"\n{HELP_TEXT}")
        if key == ord("v"):
            idx = _OVERLAY_CYCLE.index(overlay_mode)
            overlay_mode = _OVERLAY_CYCLE[(idx + 1) % len(_OVERLAY_CYCLE)]
            render_current()
            print(f"Overlay: {overlay_mode.value}")
            print(f"\n{HELP_TEXT}")

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
