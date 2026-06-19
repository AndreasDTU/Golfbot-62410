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
from path.route_strategy import (
    NearestNeighborStrategy,
    RoutePlannerInput,
    RouteStrategyResult,
)
from path.tools.pathfinding_sandbox import (
    RANDOM_BALL_COUNT,
    SandboxBall,
    SandboxState,
    balls_as_smoothed_coordinates,
    build_red_zones,
    centered_cross_obstacle_contours,
    random_ball_positions,
)
from perception.vision.debug import SchematicRenderer
from perception.vision.geometry import CoordinateMapper
from perception.vision.grid_mapping import OccupancyGridBuilder

WINDOW_NAME = "Pickup Geometry Visualizer"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_FILE = OUTPUT_DIR / "pickup_geometry.png"

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
)


# ---------------------------------------------------------------------------
# Test scenario generator
# ---------------------------------------------------------------------------

def generate_test_scenario(
    state: SandboxState,
    seed: int = 42,
) -> None:
    """Populate the sandbox with randomly placed balls and a center cross."""
    state.rng = np.random.default_rng(seed)
    state.balls.clear()
    state.obstacle_contours = centered_cross_obstacle_contours(state)
    state.next_track_id = 1

    positions = random_ball_positions(state, RANDOM_BALL_COUNT)
    orange_index = int(state.rng.integers(0, RANDOM_BALL_COUNT))

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
) -> np.ndarray:
    """Overlay pickup geometry onto an existing schematic image.

    Draws distance-field heatmap, per-ball rings, and color-coded candidates
    (green=SAFE, yellow=IN_BETWEEN, red=CONSTRAINED).
    """
    h, w = image.shape[:2]

    # --- Distance field heatmap overlay ---
    df = result.distance_field
    if df.size > 0:
        max_d = max(result.tank_turn_radius_cm * 1.5, 1.0)
        normalized = np.clip(df / max_d, 0.0, 1.0)
        # Resize to schematic size
        resized = cv2.resize(normalized.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        # Flip vertically (grid is top-left origin, schematic matches)
        heatmap_u8 = (resized * 255).astype(np.uint8)
        heatmap_bgr = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
        cv2.addWeighted(heatmap_bgr, 0.15, image, 0.85, 0, image)

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
    compile_strategy: CompileStrategy = CompileStrategy.MINIMAL,
) -> np.ndarray:
    """Generate a scenario, compute geometry + route, render, and print summary."""
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
    )

    # Render
    image = render_base_schematic(state)
    draw_pickup_geometry(image, result, state.renderer.mapper, config.field)
    draw_route_plan(image, strategy_result, result, state.renderer.mapper, config.field, start,
                    route_plan=route_plan)
    draw_start_marker(image, start, state.renderer.mapper)
    draw_legend(image, result, strategy_result)

    # Save output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUTPUT_FILE), image)

    # Print summary
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

    return image


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
        choices=["full", "minimal"],
        default="minimal",
        help="Initial compile strategy: 'full' (Douglas-Peucker) or 'minimal' (greedy visibility)",
    )
    args = parser.parse_args()

    if args.mode == "camera":
        print("Camera mode not yet implemented -- falling back to test mode.")

    config = AppConfig.from_repo_root(REPO_ROOT)
    renderer = SchematicRenderer(config.field, config.windows, config.robot)
    occupancy_builder = OccupancyGridBuilder(config.field, config.robot, renderer.mapper)
    geometry = renderer.robot_geometry_from_params(None)
    state = SandboxState(config=config, renderer=renderer)

    seed = args.seed
    strategy_index = next(
        index for index, option in enumerate(STRATEGY_OPTIONS)
        if option.key == args.strategy
    )
    compile_strategy = CompileStrategy(args.compile)
    image: np.ndarray | None = None

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, config.windows.schematic_width_px, config.windows.schematic_height_px)

    HELP_TEXT = "Press 'r' to randomize, 'c' to toggle compile strategy, Esc/q to quit."

    def render_current() -> None:
        nonlocal image
        image = _compute_and_render(
            state,
            occupancy_builder,
            geometry,
            seed,
            STRATEGY_OPTIONS[strategy_index],
            compile_strategy=compile_strategy,
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
        if key == ord("c"):
            if compile_strategy == CompileStrategy.MINIMAL:
                compile_strategy = CompileStrategy.FULL
            else:
                compile_strategy = CompileStrategy.MINIMAL
            render_current()
            print(f"\n{HELP_TEXT}")

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
