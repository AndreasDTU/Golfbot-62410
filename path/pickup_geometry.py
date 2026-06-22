"""Pickup-point geometry: ring sampling, obstacle classification, and distance field.

Given a set of balls on the field, computes for each ball the set of valid
robot-origin positions from which the pickup tube can reach the ball.  Each
candidate is classified by obstacle clearance into one of three categories:

    SAFE          — enough room for a full tank turn in place.
    IN_BETWEEN    — tube can sweep but tank turn would clip obstacles.
    CONSTRAINED   — very tight; requires a straight-line approach.

The distance field (from ``cv2.distanceTransform``) is exported so that
downstream layers can compute intermediate nodes.

This module is pure geometry.  It carries **no rendering dependency** so
the result can be consumed by both the visualizer and the path planner.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from localization.models import RobotGeometry
from path.models import PlannedBallTarget

DEFAULT_N_SAMPLES = 72

# Physical pickup pipe diameter (cm).  Used for tube sweep radius calculation.
PIPE_DIAMETER_CM = 4.5


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class PickupCategory(Enum):
    """Obstacle-clearance classification for a pickup candidate."""

    SAFE = "safe"
    IN_BETWEEN = "in_between"
    CONSTRAINED = "constrained"


@dataclass(frozen=True)
class PickupCandidate:
    """One valid robot-origin position, heading, and clearance classification."""

    x_cm: float
    y_cm: float
    theta_rad: float
    category: PickupCategory
    obstacle_distance_cm: float
    ball_index: int


@dataclass(frozen=True)
class InvalidHeading:
    """A heading that was rejected (no valid offset placed the origin in a legal cell)."""

    x_cm: float  # robot origin position (at nominal ring radius)
    y_cm: float
    theta_rad: float
    ball_index: int


@dataclass(frozen=True)
class BallCandidates:
    """Pickup candidates for a single ball."""

    ball: PlannedBallTarget
    candidates: tuple[PickupCandidate, ...]
    invalid_headings: tuple[InvalidHeading, ...]  # headings with no valid placement
    reachable: bool  # True if at least one candidate exists


@dataclass(frozen=True)
class PickupGeometryResult:
    """Complete pickup geometry for a scene."""

    balls: tuple[BallCandidates, ...]
    distance_field: np.ndarray       # full distance transform (cm per pixel)
    tank_turn_radius_cm: float
    tube_sweep_radius_cm: float
    safe_radius_cm: float            # max(tank_turn, tube_sweep) — SAFE threshold
    constrained_radius_cm: float     # min(tank_turn, tube_sweep) — CONSTRAINED threshold
    ring_radius_cm: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pickup_reach_cm(geometry: RobotGeometry) -> float:
    """The tube-tip distance from robot origin."""
    return math.hypot(geometry.tube_forward_cm, geometry.tube_right_cm)


def _origin_for_heading(
    ball_x: float,
    ball_y: float,
    heading_rad: float,
    tube_forward: float,
    tube_right: float,
) -> tuple[float, float]:
    """Compute the robot origin that places the tube tip on the ball.

    Matches the formula in ``HybridAStarPlanner.pickup_aligned_pose_for_theta``.
    """
    cos_h = math.cos(heading_rad)
    sin_h = math.sin(heading_rad)
    return (
        ball_x - cos_h * tube_forward - sin_h * tube_right,
        ball_y - sin_h * tube_forward + cos_h * tube_right,
    )


def _reach_offsets(mouth_radius_cm: float, step_cm: float = 1.0) -> list[float]:
    """Return reach offsets to try, ordered by distance from the nominal ring.

    Always starts with 0.0 (exact ring).  If mouth_radius_cm > 0, adds
    alternating +/- step offsets up to the mouth radius.
    """
    offsets = [0.0]
    if mouth_radius_cm <= 0.0:
        return offsets
    d = step_cm
    while d <= mouth_radius_cm + 1e-9:
        offsets.append(-d)
        offsets.append(d)
        d += step_cm
    return offsets


def _field_cm_to_grid(
    x_cm: float,
    y_cm: float,
    field_height_cm: float,
    grid_h: int,
    grid_w: int,
) -> tuple[int, int] | None:
    """Convert bottom-left field-cm to grid (row, col).  None if out of bounds."""
    col = round(x_cm)
    row = round(field_height_cm - y_cm)
    if 0 <= row < grid_h and 0 <= col < grid_w:
        return row, col
    return None


def _classify(
    obstacle_distance_cm: float,
    safe_radius_cm: float,
    constrained_radius_cm: float,
) -> PickupCategory:
    """Classify a candidate by its obstacle clearance.

    safe_radius_cm is the larger of (tank_turn_radius, tube_sweep_radius).
    constrained_radius_cm is the smaller.  Candidates with distance above
    the safe radius are SAFE (free tank turn + tube sweep).  Between safe
    and constrained is IN_BETWEEN.  Below constrained is CONSTRAINED.
    """
    if obstacle_distance_cm > safe_radius_cm:
        return PickupCategory.SAFE
    if obstacle_distance_cm > constrained_radius_cm:
        return PickupCategory.IN_BETWEEN
    return PickupCategory.CONSTRAINED


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------

def compute_pickup_geometry(
    field_width_cm: float,
    field_height_cm: float,
    obstacle_grid: np.ndarray,
    balls: list[PlannedBallTarget],
    geometry: RobotGeometry,
    n_samples: int = DEFAULT_N_SAMPLES,
) -> PickupGeometryResult:
    """Compute the distance field and per-ball classified pickup candidates.

    Parameters
    ----------
    field_width_cm, field_height_cm:
        Physical field dimensions in centimetres.
    obstacle_grid:
        Binary occupancy grid at 1 cm/px (0 = free, >0 = obstacle).
        Shape ``(grid_height, grid_width)`` with top-left origin.
        Includes walls and red zones.
    balls:
        Detected ball targets in field-cm coordinates.
    geometry:
        Robot body and tube geometry.
    n_samples:
        Number of evenly-spaced heading samples per ring (default 72 = 5 deg).

    Returns
    -------
    PickupGeometryResult
        Distance field, radii, and per-ball classified candidates.
    """
    grid_h, grid_w = obstacle_grid.shape[:2]

    # --- Merge wall borders into obstacle grid --------------------------------
    # The occupancy grid from the pipeline only contains red zones (cross, etc).
    # We add 1-pixel wall borders so the distance transform also measures
    # distance to field perimeter, which is critical for wall-adjacent balls.
    combined = (obstacle_grid > 0).astype(np.uint8)
    combined[0, :] = 1       # top wall  (grid top-left origin)
    combined[-1, :] = 1      # bottom wall
    combined[:, 0] = 1       # left wall
    combined[:, -1] = 1      # right wall

    # --- Distance field -------------------------------------------------------
    free_mask = (combined == 0).astype(np.uint8)
    distance_field = cv2.distanceTransform(free_mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

    # --- Radii ----------------------------------------------------------------
    R = pickup_reach_cm(geometry)
    half_width = geometry.width_cm * 0.5
    tank_turn_radius = math.hypot(geometry.rear_cm, half_width)
    tube_sweep_radius = math.hypot(geometry.tube_forward_cm, PIPE_DIAMETER_CM * 0.5)
    # Classification thresholds: safe above the larger, constrained below the smaller.
    safe_radius = max(tank_turn_radius, tube_sweep_radius)
    constrained_radius = min(tank_turn_radius, tube_sweep_radius)

    # --- Per-ball ring sampling -----------------------------------------------
    tube_fwd = geometry.tube_forward_cm
    tube_right = geometry.tube_right_cm
    mouth_r = geometry.mouth_radius_cm
    angle_step = 2.0 * math.pi / n_samples

    offsets = _reach_offsets(mouth_r)
    if R > 1e-6:
        offset_scales = [(R + off) / R for off in offsets]
    else:
        offset_scales = [1.0] * len(offsets)

    ball_results: list[BallCandidates] = []
    for ball_idx, ball in enumerate(balls):
        candidates: list[PickupCandidate] = []
        invalid: list[InvalidHeading] = []

        for i in range(n_samples):
            heading = i * angle_step
            found = False

            for off, scale in zip(offsets, offset_scales):
                fwd_s = tube_fwd * scale
                right_s = tube_right * scale
                ox, oy = _origin_for_heading(ball.x_cm, ball.y_cm, heading, fwd_s, right_s)
                cell = _field_cm_to_grid(ox, oy, field_height_cm, grid_h, grid_w)
                if cell is None:
                    continue

                # Legality check: robot origin must be at least half_width
                # from any obstacle/wall (body circle clearance).
                dist_at_origin = float(distance_field[cell[0], cell[1]])
                if dist_at_origin < half_width:
                    continue

                category = _classify(dist_at_origin, safe_radius, constrained_radius)
                candidates.append(PickupCandidate(
                    x_cm=ox,
                    y_cm=oy,
                    theta_rad=heading,
                    category=category,
                    obstacle_distance_cm=dist_at_origin,
                    ball_index=ball_idx,
                ))
                found = True
                break  # accept first valid offset for this heading

            if not found:
                # Record the nominal (offset=0) origin for visualization.
                nom_ox, nom_oy = _origin_for_heading(
                    ball.x_cm, ball.y_cm, heading, tube_fwd, tube_right,
                )
                invalid.append(InvalidHeading(
                    x_cm=nom_ox, y_cm=nom_oy,
                    theta_rad=heading, ball_index=ball_idx,
                ))

        ball_results.append(BallCandidates(
            ball=ball,
            candidates=tuple(candidates),
            invalid_headings=tuple(invalid),
            reachable=len(candidates) > 0,
        ))

    return PickupGeometryResult(
        balls=tuple(ball_results),
        distance_field=distance_field,
        tank_turn_radius_cm=tank_turn_radius,
        tube_sweep_radius_cm=tube_sweep_radius,
        safe_radius_cm=safe_radius,
        constrained_radius_cm=constrained_radius,
        ring_radius_cm=R,
    )
