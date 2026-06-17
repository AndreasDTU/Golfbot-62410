"""Pickup-point geometry: ring sampling and legal-region computation.

Given a set of balls on the field, computes for each ball the set of valid
robot-origin positions from which the pickup tube can reach the ball.  The
robot origin must lie on a ring of radius R (the tube reach) centred on the
ball, and must also fall inside the *legal origin region* — the field
boundary eroded by R minus every obstacle dilated by R.

This module is pure geometry.  It carries **no rendering dependency** so
the result can be consumed by both the visualizer and the path planner.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from localization.models import RobotGeometry
from path.pathfinding.models import PlannedBallTarget

DEFAULT_N_SAMPLES = 72


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PickupPose:
    """One valid robot-origin position and heading for picking up a ball."""

    x_cm: float
    y_cm: float
    theta_rad: float  # heading: faces the ball
    reach_offset_cm: float = 0.0  # 0 = exact ring, ±N = mouth tolerance used


@dataclass(frozen=True)
class BallPickupResult:
    """Pickup geometry result for a single ball."""

    ball_x_cm: float
    ball_y_cm: float
    track_id: int
    label: str
    ring_radius_cm: float
    valid_points: tuple[PickupPose, ...]
    invalid_angles_rad: tuple[float, ...]  # headings that fell outside legal region
    reachable: bool  # len(valid_points) > 0


@dataclass(frozen=True)
class PickupGeometryResult:
    """Complete pickup geometry for a scene."""

    legal_region_mask: np.ndarray  # bool, field-grid shape (H×W at 1 cm/px)
    eroded_field_mask: np.ndarray  # bool, field inset by R
    dilated_obstacle_mask: np.ndarray  # bool, obstacles grown by R
    ring_radius_cm: float
    mouth_radius_cm: float
    balls: tuple[BallPickupResult, ...]
    field_width_cm: float
    field_height_cm: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pickup_reach_cm(geometry: RobotGeometry) -> float:
    """The tube-tip distance from robot origin."""
    return math.hypot(geometry.tube_forward_cm, geometry.tube_right_cm)


def _circular_kernel(radius_px: int) -> np.ndarray:
    """Return a circular structuring element with the given radius in pixels."""
    diameter = 2 * radius_px + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))


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
    alternating ±step offsets up to the mouth radius.
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
    """Compute the legal origin region and per-ball valid pickup poses.

    Parameters
    ----------
    field_width_cm, field_height_cm:
        Physical field dimensions in centimetres.
    obstacle_grid:
        Binary occupancy grid at 1 cm/px (0 = free, >0 = obstacle).
        Shape ``(grid_height, grid_width)`` with top-left origin.
    balls:
        Detected ball targets in field-cm coordinates.
    geometry:
        Robot body and tube geometry.
    n_samples:
        Number of evenly-spaced heading samples per ring (default 72 = 5°).

    Returns
    -------
    PickupGeometryResult
        Legal region masks and per-ball ring/valid-point data.
    """
    grid_h, grid_w = obstacle_grid.shape[:2]
    R = pickup_reach_cm(geometry)
    kernel_radius = max(1, round(R))
    kernel = _circular_kernel(kernel_radius)

    # --- Legal origin region ---------------------------------------------------

    # Eroded field: pixels whose centre is ≥ R from any wall.
    # borderValue=0 treats the area outside the image as wall, so edge
    # pixels are correctly eroded.
    field_mask = np.ones((grid_h, grid_w), dtype=np.uint8)
    eroded_field = cv2.erode(
        field_mask, kernel, iterations=1,
        borderType=cv2.BORDER_CONSTANT, borderValue=0,
    )

    # Dilated obstacles: pixels within R of any obstacle cell.
    binary_obstacles = (obstacle_grid > 0).astype(np.uint8)
    dilated_obstacles = cv2.dilate(binary_obstacles, kernel, iterations=1)

    legal_region = (eroded_field > 0) & (dilated_obstacles == 0)

    # --- Per-ball ring sampling ------------------------------------------------

    tube_fwd = geometry.tube_forward_cm
    tube_right = geometry.tube_right_cm
    mouth_r = geometry.mouth_radius_cm
    angle_step = 2.0 * math.pi / n_samples

    # Pre-compute reach offsets: [0.0, -1.0, +1.0, -2.0, +2.0, ...] up to mouth_r.
    # Exact ring (offset 0) is always tried first to preserve precision.
    offsets = _reach_offsets(mouth_r)

    # Scale factors for each offset: how to stretch tube_fwd/tube_right so the
    # tube tip moves to R + offset along the reach direction.
    if R > 1e-6:
        offset_scales = [(R + off) / R for off in offsets]
    else:
        offset_scales = [1.0] * len(offsets)

    ball_results: list[BallPickupResult] = []
    for ball in balls:
        valid: list[PickupPose] = []
        invalid: list[float] = []

        for i in range(n_samples):
            heading = i * angle_step
            found = False
            for off, scale in zip(offsets, offset_scales):
                fwd_s = tube_fwd * scale
                right_s = tube_right * scale
                ox, oy = _origin_for_heading(ball.x_cm, ball.y_cm, heading, fwd_s, right_s)
                cell = _field_cm_to_grid(ox, oy, field_height_cm, grid_h, grid_w)
                if cell is not None and legal_region[cell[0], cell[1]]:
                    valid.append(PickupPose(x_cm=ox, y_cm=oy, theta_rad=heading, reach_offset_cm=off))
                    found = True
                    break
            if not found:
                invalid.append(heading)

        ball_results.append(
            BallPickupResult(
                ball_x_cm=ball.x_cm,
                ball_y_cm=ball.y_cm,
                track_id=ball.track_id,
                label=ball.label,
                ring_radius_cm=R,
                valid_points=tuple(valid),
                invalid_angles_rad=tuple(invalid),
                reachable=len(valid) > 0,
            )
        )

    return PickupGeometryResult(
        legal_region_mask=legal_region,
        eroded_field_mask=eroded_field.astype(bool),
        dilated_obstacle_mask=dilated_obstacles.astype(bool),
        ring_radius_cm=R,
        mouth_radius_cm=mouth_r,
        balls=tuple(ball_results),
        field_width_cm=field_width_cm,
        field_height_cm=field_height_cm,
    )
