"""Pickup-point geometry visualizer.

Computes and renders the pickup-ring geometric model for every ball on the
field.  Two input modes:

    --mode=test     Deterministic test scenario with edge-case ball placements
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
from path.pathfinding.models import HybridPose, PlannedBallTarget
from path.pickup_geometry import (
    BallPickupResult,
    PickupGeometryResult,
    compute_pickup_geometry,
    pickup_reach_cm,
)
from path.route_strategy import (
    ObstacleGeometry,
    RouteEdge,
    RoutePlannerInput,
    RouteStrategyResult,
)
from path.route_v1 import IntersectionPriorityStrategy, SetCoverNearestNeighborStrategy
from path.tools.pathfinding_sandbox import (
    RANDOM_BALL_COUNT,
    RANDOM_WHITE_BALL_COUNT,
    SandboxBall,
    SandboxState,
    balls_as_smoothed_coordinates,
    build_red_zones,
    centered_cross_obstacle_contours,
    random_ball_positions,
    red_zone_from_contour,
)
from perception.vision.debug import SchematicRenderer
from perception.vision.geometry import CoordinateMapper
from perception.vision.grid_mapping import OccupancyGridBuilder

WINDOW_NAME = "Pickup Geometry Visualizer"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_FILE = OUTPUT_DIR / "pickup_geometry.png"

# Cross geometry (same as pathfinding_sandbox)
CENTER_CROSS_SIZE_CM = 20.0
CENTER_CROSS_ARM_WIDTH_CM = 3.0


class StrategyOption:
    """One selectable visualizer routing strategy."""

    def __init__(self, key: str, label: str, strategy_cls) -> None:
        self.key = key
        self.label = label
        self.strategy_cls = strategy_cls

    def create(self):
        return self.strategy_cls()


STRATEGY_OPTIONS = (
    StrategyOption("set-cover", "Set-cover nearest-neighbor", SetCoverNearestNeighborStrategy),
    StrategyOption("intersections", "Intersection-priority", IntersectionPriorityStrategy),
)


# ---------------------------------------------------------------------------
# Test scenario generator
# ---------------------------------------------------------------------------

def generate_test_scenario(
    state: SandboxState,
    seed: int = 42,
) -> None:
    """Populate the sandbox with randomly placed balls and a center cross.

    Uses ``random_ball_positions`` from the pathfinding sandbox for
    rejection-sampled placement (min spacing, cross clearance, field margin).
    The *seed* makes every run reproducible; pass different seeds via
    ``--seed`` to explore different layouts.
    """
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

def _mask_to_schematic(
    mask: np.ndarray,
    field_config: FieldConfig,
    schematic_w: int,
    schematic_h: int,
) -> np.ndarray:
    """Resize a field-grid boolean mask to schematic pixel dimensions."""
    uint_mask = (mask > 0).astype(np.uint8) * 255
    return cv2.resize(uint_mask, (schematic_w, schematic_h), interpolation=cv2.INTER_NEAREST)


def draw_pickup_geometry(
    image: np.ndarray,
    result: PickupGeometryResult,
    mapper: CoordinateMapper,
    field_config: FieldConfig,
) -> np.ndarray:
    """Overlay pickup geometry onto an existing schematic image.

    Draws forbidden zones, per-ball rings, valid/invalid sample points,
    and unreachable indicators.  Returns the annotated image (modified in
    place for performance, but also returned for convenience).
    """
    h, w = image.shape[:2]

    # --- Forbidden wall band (eroded region complement) ---------------------
    eroded_schematic = _mask_to_schematic(result.eroded_field_mask, field_config, w, h)
    wall_band = eroded_schematic == 0  # pixels OUTSIDE the eroded field
    overlay = image.copy()
    overlay[wall_band] = (0, 0, 160)  # dark red
    cv2.addWeighted(overlay, 0.30, image, 0.70, 0, image)

    # --- Dilated obstacle zone ----------------------------------------------
    dilated_schematic = _mask_to_schematic(result.dilated_obstacle_mask, field_config, w, h)
    inflation_zone = dilated_schematic > 0
    overlay = image.copy()
    overlay[inflation_zone] = (0, 60, 180)  # orange-red
    cv2.addWeighted(overlay, 0.25, image, 0.75, 0, image)

    # --- Per-ball rings and sample points -----------------------------------
    for ball_result in result.balls:
        ball_px = mapper.field_metric_cm_to_schematic(
            (ball_result.ball_x_cm, ball_result.ball_y_cm)
        )
        ring_radius_px = _cm_to_px(result.ring_radius_cm, field_config, w)

        # Ring circle
        ring_color = (255, 200, 0) if ball_result.reachable else (0, 0, 200)
        cv2.circle(image, ball_px, ring_radius_px, ring_color, 1, cv2.LINE_AA)

        # Valid sample points — green filled dots (exact ring) or yellow (offset)
        for pose in ball_result.valid_points:
            pt = mapper.field_metric_cm_to_schematic((pose.x_cm, pose.y_cm))
            if pose.reach_offset_cm == 0.0:
                cv2.circle(image, pt, 3, (0, 220, 0), -1, cv2.LINE_AA)
            else:
                cv2.circle(image, pt, 3, (0, 200, 255), -1, cv2.LINE_AA)  # yellow-orange

        # Invalid sample points — dim red hollow dots
        tube_fwd = result.ring_radius_cm  # approximate for rendering
        tube_right = 0.0
        for heading in ball_result.invalid_angles_rad:
            cos_h = math.cos(heading)
            sin_h = math.sin(heading)
            ox = ball_result.ball_x_cm - cos_h * tube_fwd - sin_h * tube_right
            oy = ball_result.ball_y_cm - sin_h * tube_fwd + cos_h * tube_right
            pt = mapper.field_metric_cm_to_schematic((ox, oy))
            cv2.circle(image, pt, 2, (0, 0, 140), 1, cv2.LINE_AA)

        # Unreachable: red X over ball
        if not ball_result.reachable:
            size = 8
            cv2.line(
                image,
                (ball_px[0] - size, ball_px[1] - size),
                (ball_px[0] + size, ball_px[1] + size),
                (0, 0, 255), 2, cv2.LINE_AA,
            )
            cv2.line(
                image,
                (ball_px[0] - size, ball_px[1] + size),
                (ball_px[0] + size, ball_px[1] - size),
                (0, 0, 255), 2, cv2.LINE_AA,
            )

    # --- Legend text ---------------------------------------------------------
    reachable_count = sum(1 for b in result.balls if b.reachable)
    total = len(result.balls)
    lines = [
        f"R={result.ring_radius_cm:.1f}cm  mouth={result.mouth_radius_cm:.1f}cm  "
        f"n={len(result.balls[0].valid_points) + len(result.balls[0].invalid_angles_rad) if result.balls else 0}/ring  "
        f"Balls: {reachable_count}/{total} reachable",
    ]
    y_px = h - 20
    for line in lines:
        cv2.putText(image, line, (12, y_px), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, line, (12, y_px), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y_px += 22

    return image


def _cm_to_px(cm: float, field_config: FieldConfig, schematic_width_px: int) -> int:
    """Convert a distance in centimetres to schematic pixels."""
    return max(1, int(round(cm * (schematic_width_px - 1) / max(1.0, field_config.width_cm))))


def draw_route_plan(
    image: np.ndarray,
    strategy_result: RouteStrategyResult,
    geometry_result: PickupGeometryResult,
    mapper: CoordinateMapper,
    field_config: FieldConfig,
) -> np.ndarray:
    """Overlay route plan visualization onto an existing schematic image."""
    h, w = image.shape[:2]
    cover_points = strategy_result.cover_points
    ordered = strategy_result.ordered_indices
    edges = strategy_result.edges
    route_plan = strategy_result.route_plan

    def to_px(pose: HybridPose) -> tuple[int, int]:
        return mapper.field_metric_cm_to_schematic((pose.x_cm, pose.y_cm))

    # Build the set of (from, to) graph-index pairs actually used in the route.
    route_edge_keys: set[tuple[int, int]] = set()
    prev_graph = 0  # start node
    for cp_idx in ordered:
        graph_idx = cp_idx + 1
        route_edge_keys.add((prev_graph, graph_idx))
        prev_graph = graph_idx
    # Unload leg
    if route_plan.unload_pose is not None:
        unload_graph_idx = len(cover_points) + 1
        route_edge_keys.add((prev_graph, unload_graph_idx))

    edge_lookup: dict[tuple[int, int], RouteEdge] = {
        (e.from_index, e.to_index): e for e in edges
    }

    # --- Layer 1: Blocked route edges (thin red dashed) ---
    for key in route_edge_keys:
        edge = edge_lookup.get(key)
        if edge is None or not edge.blocked:
            continue
        from_pose = _graph_index_to_pose(
            edge.from_index, route_plan.points[0], cover_points, route_plan.unload_pose)
        to_pose = _graph_index_to_pose(
            edge.to_index, route_plan.points[0], cover_points, route_plan.unload_pose)
        if from_pose is not None and to_pose is not None:
            _draw_dotted_line(image, to_px(from_pose), to_px(to_pose), (0, 0, 180), 1, gap=8)

    # --- Layer 2: Coverage lines (thin gray dotted) ---
    for cp in cover_points:
        cp_px = to_px(cp.pose)
        for ball_idx in cp.covered_ball_indices:
            ball = geometry_result.balls[ball_idx]
            ball_px = mapper.field_metric_cm_to_schematic((ball.ball_x_cm, ball.ball_y_cm))
            _draw_dotted_line(image, cp_px, ball_px, (180, 180, 180), 1)

    # --- Layer 3: Detour waypoints used in the route (yellow diamonds) ---
    for key in route_edge_keys:
        edge = edge_lookup.get(key)
        if edge is not None:
            detour_points = edge.detour_waypoints
            if not detour_points and edge.detour_waypoint is not None:
                detour_points = (edge.detour_waypoint,)
            for waypoint in detour_points:
                wp_px = to_px(waypoint)
                _draw_diamond(image, wp_px, 6, (0, 220, 255))

    # --- Layer 4: Final route polyline (thick cyan with arrowheads) ---
    route_px = [to_px(p) for p in route_plan.points]
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
            right = (mid_x - int(ux * arrow_size + uy * arrow_size * 0.5),
                     mid_y - int(uy * arrow_size - ux * arrow_size * 0.5))
            cv2.fillPoly(image, [np.array([tip, left, right], dtype=np.int32)], (255, 220, 0))

    # --- Layer 5: Cover point markers (filled cyan circles with order numbers) ---
    for visit_order, cp_idx in enumerate(ordered):
        cp = cover_points[cp_idx]
        px = to_px(cp.pose)
        if len(cp.covered_ball_indices) > 1:
            cv2.circle(image, px, 13, (255, 0, 255), 2, cv2.LINE_AA)
        cv2.circle(image, px, 8, (255, 220, 0), -1, cv2.LINE_AA)
        cv2.circle(image, px, 8, (0, 0, 0), 1, cv2.LINE_AA)
        label = str(visit_order + 1)
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        tx = px[0] - text_size[0] // 2
        ty = px[1] + text_size[1] // 2
        cv2.putText(image, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

    # Start pose marker (green square)
    start_px = to_px(route_plan.points[0])
    cv2.rectangle(image,
                  (start_px[0] - 5, start_px[1] - 5),
                  (start_px[0] + 5, start_px[1] + 5),
                  (0, 255, 0), -1, cv2.LINE_AA)
    cv2.rectangle(image,
                  (start_px[0] - 5, start_px[1] - 5),
                  (start_px[0] + 5, start_px[1] + 5),
                  (0, 0, 0), 1, cv2.LINE_AA)

    # Unload pose marker (magenta triangle)
    if route_plan.unload_pose is not None:
        up = to_px(route_plan.unload_pose)
        tri = np.array([
            [up[0], up[1] - 7],
            [up[0] + 6, up[1] + 5],
            [up[0] - 6, up[1] + 5],
        ], dtype=np.int32)
        cv2.fillPoly(image, [tri], (255, 0, 255))
        cv2.polylines(image, [tri], True, (0, 0, 0), 1, cv2.LINE_AA)

    # Unload goal marker (small magenta X at the goal opening)
    if route_plan.unload_goal_cm is not None:
        gx, gy = route_plan.unload_goal_cm
        goal_px = mapper.field_metric_cm_to_schematic((gx, gy))
        sz = 5
        cv2.line(image, (goal_px[0] - sz, goal_px[1] - sz),
                 (goal_px[0] + sz, goal_px[1] + sz), (255, 0, 255), 2, cv2.LINE_AA)
        cv2.line(image, (goal_px[0] - sz, goal_px[1] + sz),
                 (goal_px[0] + sz, goal_px[1] - sz), (255, 0, 255), 2, cv2.LINE_AA)

    # --- Layer 6: Legend ---
    total_covered = sum(len(cp.covered_ball_indices) for cp in cover_points)
    used_detours = sum(
        1 for key in route_edge_keys
        if (e := edge_lookup.get(key)) is not None and e.blocked and e.detour_waypoint is not None
    )
    has_unload = route_plan.unload_pose is not None
    legend = (f"Route: {len(cover_points)} stops, {total_covered} balls covered, "
              f"{used_detours} detours" + (", +unload" if has_unload else ""))
    y_legend = h - 42
    cv2.putText(image, legend, (12, y_legend), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, legend, (12, y_legend), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 220, 0), 1, cv2.LINE_AA)

    return image


def draw_strategy_label(
    image: np.ndarray,
    strategy_option: StrategyOption,
) -> None:
    """Draw the currently selected routing strategy in the schematic."""
    text = f"Strategy: {strategy_option.label}"
    x_px = 12
    y_px = 88
    cv2.putText(image, text, (x_px, y_px), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, (x_px, y_px), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def _graph_index_to_pose(
    idx: int,
    start_pose: HybridPose,
    cover_points: tuple,
    unload_pose: HybridPose | None = None,
) -> HybridPose | None:
    """Map graph node index to the corresponding HybridPose."""
    if idx == 0:
        return start_pose
    cp_idx = idx - 1
    if 0 <= cp_idx < len(cover_points):
        return cover_points[cp_idx].pose
    # Unload node sits at index len(cover_points) + 1
    if unload_pose is not None and idx == len(cover_points) + 1:
        return unload_pose
    return None


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


def _draw_diamond(
    image: np.ndarray,
    center: tuple[int, int],
    size: int,
    color: tuple[int, int, int],
) -> None:
    """Draw a filled diamond marker."""
    pts = np.array([
        [center[0], center[1] - size],
        [center[0] + size, center[1]],
        [center[0], center[1] + size],
        [center[0] - size, center[1]],
    ], dtype=np.int32)
    cv2.fillPoly(image, [pts], color)
    cv2.polylines(image, [pts], True, (0, 0, 0), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Base schematic rendering (reuses existing renderer)
# ---------------------------------------------------------------------------

def render_base_schematic(
    state: SandboxState,
) -> np.ndarray:
    """Draw the field, obstacles, and balls using the existing schematic renderer."""
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
) -> np.ndarray:
    """Generate a scenario, compute geometry + route, render, and print summary.

    Returns the rendered image.
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

    # Compute route plan
    R = pickup_reach_cm(geometry)
    obstacle = ObstacleGeometry(
        center_x_cm=config.field.width_cm * 0.5,
        center_y_cm=config.field.height_cm * 0.5,
        half_size_cm=CENTER_CROSS_SIZE_CM * 0.5,
        half_arm_width_cm=CENTER_CROSS_ARM_WIDTH_CM * 0.5,
    )
    # Hardcoded robot initial position for test mode
    start = HybridPose(x_cm=20.0, y_cm=20.0, theta_rad=0.0)
    # Unload: staging pose perpendicular to the left-side goal opening
    # Matches HybridAStarPlanner.small_goal_unload_pose()
    unload_goal_cm = (0.0, config.field.height_cm * 0.5)
    unload_reach_cm = geometry.rear_cm + geometry.unload_extension_cm
    unload_pose = HybridPose(
        x_cm=unload_reach_cm + 2.0,  # +2 cm staging margin
        y_cm=config.field.height_cm * 0.5,
        theta_rad=0.0,
    )
    route_input = RoutePlannerInput(
        geometry_result=result,
        obstacle=obstacle,
        start_pose=start,
        robot_radius_cm=R,
        field_width_cm=config.field.width_cm,
        field_height_cm=config.field.height_cm,
        unload_pose=unload_pose,
        unload_goal_cm=unload_goal_cm,
    )
    strategy = strategy_option.create()
    strategy_result = strategy.plan(route_input)

    # Render
    image = render_base_schematic(state)
    draw_pickup_geometry(image, result, state.renderer.mapper, config.field)
    draw_route_plan(image, strategy_result, result, state.renderer.mapper, config.field)
    draw_strategy_label(image, strategy_option)

    # Save output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUTPUT_FILE), image)

    # Print summary
    print(f"\n--- seed={seed} strategy={strategy_option.key} ---")
    print(f"R = {R:.1f} cm  Field: {config.field.width_cm} x {config.field.height_cm} cm")
    print(f"Balls: {len(result.balls)}")
    for b in result.balls:
        if b.reachable:
            offset_count = sum(1 for p in b.valid_points if p.reach_offset_cm != 0.0)
            status = f"{len(b.valid_points)} valid"
            if offset_count:
                status += f" ({offset_count} via mouth tolerance)"
        else:
            status = "UNREACHABLE"
        print(f"  #{b.track_id} ({b.label}) at ({b.ball_x_cm:.1f}, {b.ball_y_cm:.1f}): {status}")

    # Route summary
    total_covered = sum(len(cp.covered_ball_indices) for cp in strategy_result.cover_points)
    edge_lookup = {(e.from_index, e.to_index): e for e in strategy_result.edges}
    route_keys: list[tuple[int, int]] = []
    prev_g = 0
    for ci in strategy_result.ordered_indices:
        route_keys.append((prev_g, ci + 1))
        prev_g = ci + 1
    if strategy_result.route_plan.unload_pose is not None:
        route_keys.append((prev_g, len(strategy_result.cover_points) + 1))
    route_blocked = sum(1 for k in route_keys if (e := edge_lookup.get(k)) is not None and e.blocked)
    route_detours = sum(
        1 for k in route_keys
        if (e := edge_lookup.get(k)) is not None and e.blocked and e.detour_waypoint is not None
    )
    print(f"Route: {len(strategy_result.cover_points)} stops, "
          f"{total_covered} balls covered, "
          f"{route_blocked} blocked legs, {route_detours} detours")
    for i, cp_idx in enumerate(strategy_result.ordered_indices):
        cp = strategy_result.cover_points[cp_idx]
        balls_str = ", ".join(f"#{result.balls[bi].track_id}" for bi in cp.covered_ball_indices)
        pickup_count = len(cp.pickup_poses or (cp.pose,))
        action_text = f", {pickup_count} pickup actions" if pickup_count > 1 else ""
        print(f"  Stop {i + 1}: ({cp.pose.x_cm:.1f}, {cp.pose.y_cm:.1f}) covers [{balls_str}]{action_text}")
    rp = strategy_result.route_plan
    if rp.unload_pose is not None:
        print(f"  Unload: ({rp.unload_pose.x_cm:.1f}, {rp.unload_pose.y_cm:.1f}) -> goal {rp.unload_goal_cm}")

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
    args = parser.parse_args()

    if args.mode == "camera":
        print("Camera mode not yet implemented — falling back to test mode.")

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
    image: np.ndarray | None = None

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, config.windows.schematic_width_px, config.windows.schematic_height_px)
    cv2.createTrackbar("Strategy", WINDOW_NAME, strategy_index, len(STRATEGY_OPTIONS) - 1, lambda _value: None)

    def render_current() -> None:
        nonlocal image
        image = _compute_and_render(
            state,
            occupancy_builder,
            geometry,
            seed,
            STRATEGY_OPTIONS[strategy_index],
        )
        cv2.imshow(WINDOW_NAME, image)

    render_current()
    print("\nUse the Strategy trackbar or press 't' to switch. Press 'r' to randomize, Esc/q to quit.")

    while True:
        trackbar_index = cv2.getTrackbarPos("Strategy", WINDOW_NAME)
        if 0 <= trackbar_index < len(STRATEGY_OPTIONS) and trackbar_index != strategy_index:
            strategy_index = trackbar_index
            render_current()

        key = cv2.waitKey(50) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord("t"):
            strategy_index = (strategy_index + 1) % len(STRATEGY_OPTIONS)
            cv2.setTrackbarPos("Strategy", WINDOW_NAME, strategy_index)
            render_current()
        if key == ord("r"):
            seed += 1
            render_current()
            print("\nUse the Strategy trackbar or press 't' to switch. Press 'r' to randomize, Esc/q to quit.")

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
