"""Route-planning test with hardcoded balls in each corner.

Reuses the existing pickup visualizer infrastructure to show how the
planner handles balls placed at field extremes (near walls).

Usage:
    python path/tools/corner_ball_test.py
    python path/tools/corner_ball_test.py --extra-center  # add a ball near the cross
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

from config import AppConfig
from path.models import HybridPose, PlannedBallTarget, WaypointKind
from path.pickup_geometry import (
    PickupCategory,
    compute_pickup_geometry,
    pickup_reach_cm,
)
from path.planner import CompileStrategy, compile_route
from path.route_stats import compute_route_stats
from path.route_strategy import (
    IntersectionPriorityStrategy,
    RoutePlannerInput,
)
from path.tools.pathfinding_sandbox import (
    SandboxBall,
    SandboxState,
    build_red_zones,
    centered_cross_obstacle_contours,
)
from path.tools.pickup_visualizer import (
    OverlayMode,
    balls_as_planned_targets,
    draw_pickup_geometry,
    draw_route_plan,
    draw_start_marker,
    draw_legend,
    render_base_schematic,
    StrategyOption,
    STRATEGY_OPTIONS,
)
from perception.vision.debug import SchematicRenderer
from perception.vision.grid_mapping import OccupancyGridBuilder

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_FILE = OUTPUT_DIR / "corner_ball_test.png"
WINDOW_NAME = "Corner Ball Test"


def place_corner_balls(
    state: SandboxState,
    extra_center: bool = False,
) -> None:
    """Place one ball in each corner (orange in bottom-left) plus optional extras."""
    field = state.config.field
    state.balls.clear()
    state.obstacle_contours = centered_cross_obstacle_contours(state)
    state.next_track_id = 1

    # Ball radius (~2 cm) keeps the center off the actual wall pixel
    ball_r = field.ball_height_cm
    positions: list[tuple[float, float, str]] = [
        (ball_r, ball_r, "orange"),                                          # bottom-left corner
        (field.width_cm - ball_r, ball_r, "white"),                          # bottom-right corner
        (ball_r, field.height_cm - ball_r, "white"),                         # top-left corner
        (field.width_cm - ball_r, field.height_cm - ball_r, "white"),        # top-right corner
    ]

    if extra_center:
        # Ball near the cross (offset so it's not on top of it)
        cx = field.width_cm * 0.5 + 15.0
        cy = field.height_cm * 0.5 + 15.0
        positions.append((cx, cy, "white"))

    for x_cm, y_cm, label in positions:
        state.balls.append(
            SandboxBall(
                track_id=state.next_track_id,
                label=label,
                x_cm=x_cm,
                y_cm=y_cm,
            )
        )
        state.next_track_id += 1


def run_test(
    state: SandboxState,
    occupancy_builder: OccupancyGridBuilder,
    geometry,
    extra_center: bool = False,
    compile_strategy: CompileStrategy = CompileStrategy.SINGLE_NODE,
    overlay_mode: OverlayMode = OverlayMode.HEATMAP,
) -> np.ndarray:
    """Place corner balls, plan a route, render, and print diagnostics."""
    config = state.config

    place_corner_balls(state, extra_center=extra_center)

    # Build occupancy grid from the cross obstacle
    red_zones = build_red_zones(state)
    raw_grid = occupancy_builder.build(state.frame_shape, red_zones, dilate_for_legacy=False)

    # Compute pickup geometry (Layer 1)
    targets = balls_as_planned_targets(state)
    result = compute_pickup_geometry(
        field_width_cm=config.field.width_cm,
        field_height_cm=config.field.height_cm,
        obstacle_grid=raw_grid,
        balls=targets,
        geometry=geometry,
    )

    # Robot starts right side, centered, facing left (toward unload wall)
    unload_reach = geometry.rear_cm + geometry.unload_extension_cm
    unload_pos = (unload_reach + 2.0, config.field.height_cm * 0.5)
    start = HybridPose(
        x_cm=config.field.width_cm - geometry.rear_cm - 2.0,
        y_cm=config.field.height_cm * 0.5,
        theta_rad=math.pi,
    )

    # Route strategy (Layer 2)
    route_input = RoutePlannerInput(
        geometry_result=result,
        start_pose=start,
        field_width_cm=config.field.width_cm,
        field_height_cm=config.field.height_cm,
        unload_position=unload_pos,
    )
    strategy = IntersectionPriorityStrategy()
    strategy_result = strategy.plan(route_input)

    # Compile route (Layer 3)
    route_plan = compile_route(
        strategy_result,
        distance_field=result.distance_field,
        start_pose=start,
        half_width_cm=geometry.width_cm * 0.5,
        field_height_cm=config.field.height_cm,
        strategy=compile_strategy,
        balls=targets,
        obstacle_margin_cm=config.field.obstacle_margin_cm,
    )

    # Render
    image = render_base_schematic(state)
    draw_pickup_geometry(
        image, result, state.renderer.mapper, config.field,
        overlay_mode=overlay_mode,
        half_width_cm=geometry.width_cm * 0.5,
    )
    strategy_option = StrategyOption("intersection", "Intersection priority", IntersectionPriorityStrategy)
    draw_route_plan(image, strategy_result, result, state.renderer.mapper, config.field, start,
                    route_plan=route_plan)
    draw_start_marker(image, start, state.renderer.mapper)
    draw_legend(image, result, strategy_result,
                strategy_label=strategy_option.label,
                compile_label=f"compile: {compile_strategy.value}")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUTPUT_FILE), image)

    # Print diagnostics
    stats = compute_route_stats(route_plan)
    R = pickup_reach_cm(geometry)
    print(f"\n{'=' * 60}")
    print(f"CORNER BALL TEST — extra_center={extra_center}")
    print(f"{'=' * 60}")
    print(f"Field: {config.field.width_cm} x {config.field.height_cm} cm")
    print(f"R = {R:.1f} cm  safe_radius = {result.safe_radius_cm:.1f} cm")
    print(f"Obstacle margin: {config.field.obstacle_margin_cm} cm")
    print()

    # Per-ball info
    print("Balls:")
    for bc in result.balls:
        ball = bc.ball
        if bc.reachable:
            safe = sum(1 for c in bc.candidates if c.category == PickupCategory.SAFE)
            between = sum(1 for c in bc.candidates if c.category == PickupCategory.IN_BETWEEN)
            constrained = sum(1 for c in bc.candidates if c.category == PickupCategory.CONSTRAINED)
            status = f"{len(bc.candidates)} candidates ({safe}S/{between}B/{constrained}C)"
        else:
            status = "UNREACHABLE"
        print(f"  #{ball.track_id} ({ball.label:6s}) at ({ball.x_cm:6.1f}, {ball.y_cm:6.1f}): {status}")
    print()

    # Route strategy stops
    print("Route stops:")
    for i, stop in enumerate(strategy_result.stops):
        ball = result.balls[stop.ball_index].ball
        cat = stop.candidate.category.value
        inter_str = ""
        if stop.intermediate_node is not None:
            inter = stop.intermediate_node
            inter_str = f" via intermediate ({inter.x_cm:.1f}, {inter.y_cm:.1f})"
        print(f"  Stop {i + 1}: #{ball.track_id} ({ball.label:6s}) at "
              f"({stop.candidate.x_cm:.1f}, {stop.candidate.y_cm:.1f}) [{cat}]{inter_str}")
    print()

    # Compiled waypoints (the key diagnostic)
    print("Compiled waypoints:")
    for i, wp in enumerate(route_plan.waypoints):
        kind_str = wp.kind.value
        ball_str = f" ball_index={wp.ball_index}" if wp.ball_index is not None else ""
        constrained_str = " CONSTRAINED" if wp.obstacle_constrained else ""
        print(f"  {i:3d}: ({wp.x_cm:6.1f}, {wp.y_cm:6.1f}) {kind_str:10s}{ball_str}{constrained_str}")
    print()

    nav_count = sum(1 for w in route_plan.waypoints if w.kind == WaypointKind.NAVIGATE)
    pickup_count = sum(1 for w in route_plan.waypoints if w.kind == WaypointKind.PICKUP)
    unload_count = sum(1 for w in route_plan.waypoints if w.kind == WaypointKind.UNLOAD)
    print(f"Summary: {len(route_plan.waypoints)} waypoints "
          f"({nav_count} navigate, {pickup_count} pickup, {unload_count} unload)")
    print(f"Stats: distance={stats.total_distance_cm:.0f} cm  turns={stats.turn_count}  "
          f"max_turn={stats.max_turn_deg:.0f} deg")
    print(f"\nSaved to: {OUTPUT_FILE}")

    return image


def main() -> int:
    parser = argparse.ArgumentParser(description="Corner ball route-planning test")
    parser.add_argument("--extra-center", action="store_true",
                        help="Add a fifth ball near the center cross")
    parser.add_argument("--compile", choices=["single_node", "full", "minimal"],
                        default="single_node",
                        help="Compile strategy (default: single_node)")
    parser.add_argument("--overlay", choices=["heatmap", "collision", "none"],
                        default="heatmap",
                        help="Background overlay mode (default: heatmap)")
    parser.add_argument("--no-gui", action="store_true",
                        help="Save image and exit without showing window")
    args = parser.parse_args()

    config = AppConfig.from_repo_root(REPO_ROOT)
    renderer = SchematicRenderer(config.field, config.windows, config.robot)
    occupancy_builder = OccupancyGridBuilder(config.field, config.robot, renderer.mapper)

    # Load geometry from calibration file
    geo_params: dict[str, object] = {}
    cal_path = config.paths.robot_calibration_file
    if cal_path is not None and cal_path.exists():
        import json
        with cal_path.open() as f:
            cal = json.load(f)
        from localization.localization import RobotCalibrationCollector
        RobotCalibrationCollector.apply_geometry_to_params(geo_params, cal.get("geometry"))
    geometry = renderer.robot_geometry_from_params(geo_params or None)

    state = SandboxState(config=config, renderer=renderer)
    compile_strategy = CompileStrategy(args.compile)
    overlay_mode = OverlayMode(args.overlay)

    image = run_test(
        state, occupancy_builder, geometry,
        extra_center=args.extra_center,
        compile_strategy=compile_strategy,
        overlay_mode=overlay_mode,
    )

    if not args.no_gui:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME,
                         config.windows.schematic_width_px,
                         config.windows.schematic_height_px)
        cv2.imshow(WINDOW_NAME, image)
        print("\nPress any key to exit.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
