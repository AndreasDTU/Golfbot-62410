"""Shared data models for top-down perception.

These classes intentionally carry no frame-loop orchestration.  They are kept
small so the legacy detector can import them without changing behavior while
the larger refactor proceeds in controlled steps.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class CalibrationState(Enum):
    """Current top-down calibration mode."""

    NEEDS_CALIBRATION = "needs_calibration"
    CALIBRATED_AUTO = "calibrated_auto"
    CALIBRATING_MANUAL = "calibrating_manual"
    CALIBRATED_MANUAL = "calibrated_manual"


@dataclass(frozen=True)
class ParallaxConfig:
    """Geometry needed to project elevated points onto the ground plane."""

    marker_height_cm: float
    camera_height_cm: float
    calibration_plane_height_cm: float
    camera_center: np.ndarray


@dataclass(frozen=True)
class CameraGroundProjection:
    """Camera ground projection in the warped top-down pixel plane."""

    principal_point_px: np.ndarray
    camera_center_px: np.ndarray


@dataclass(frozen=True)
class BallDetection:
    """Ball-like object found in the frame."""

    label: str
    center: tuple[int, int]
    corrected_center: tuple[int, int]
    radius_px: int
    contour: np.ndarray
    area: float
    circularity: float


@dataclass(frozen=True)
class RedZoneDetection:
    """Detected red avoidance geometry."""

    contour: np.ndarray
    corrected_contour: np.ndarray
    bounding_box: tuple[int, int, int, int]
    center: tuple[int, int]
    corrected_center: tuple[int, int]
    area: float


@dataclass(frozen=True)
class RedCrossSpec:
    """Manually placed central red cross obstacle, defined in field centimeters.

    The cross is a plus shape: two arms of equal ``length_cm`` (tip-to-tip)
    crossing at right angles, each ``arm_width_cm`` wide.  ``angle_rad`` is the
    orientation of the primary arm axis (0 = field +X), so the geometry is fully
    determined by center + angle + the two fixed dimensions.
    """

    center_cm: tuple[float, float]
    angle_rad: float
    length_cm: float = 20.0
    arm_width_cm: float = 3.0

    @property
    def center_x_cm(self) -> float:
        """Center X coordinate, matching the planner obstacle interface."""
        return self.center_cm[0]

    @property
    def center_y_cm(self) -> float:
        """Center Y coordinate, matching the planner obstacle interface."""
        return self.center_cm[1]

    @property
    def half_size_cm(self) -> float:
        """Half of the full tip-to-tip cross length."""
        return self.length_cm / 2.0

    @property
    def half_arm_width_cm(self) -> float:
        """Half of the cross arm width."""
        return self.arm_width_cm / 2.0

    def polygon_cm(self) -> list[tuple[float, float]]:
        """Return the 12 plus-shape vertices in field cm, counter-clockwise."""
        half_len = self.length_cm / 2.0
        half_w = self.arm_width_cm / 2.0
        cos_a = math.cos(self.angle_rad)
        sin_a = math.sin(self.angle_rad)
        # Local plus outline (x' along the arm axis, y' perpendicular).
        local = [
            (half_len, -half_w), (half_len, half_w),
            (half_w, half_w), (half_w, half_len),
            (-half_w, half_len), (-half_w, half_w),
            (-half_len, half_w), (-half_len, -half_w),
            (-half_w, -half_w), (-half_w, -half_len),
            (half_w, -half_len), (half_w, -half_w),
        ]
        cx, cy = self.center_cm
        return [
            (cx + x * cos_a - y * sin_a, cy + x * sin_a + y * cos_a)
            for x, y in local
        ]

    def contains_point(self, point_cm: tuple[float, float], margin_cm: float = 0.0) -> bool:
        """True when ``point_cm`` is inside the plus, or within ``margin_cm`` of its edge.

        Used for the robot-vs-cross hitbox test: inflating the cross by half the
        robot width and testing the robot center is the Minkowski-sum way of
        asking whether the disc-approximated robot overlaps the cross.
        """
        px, py = float(point_cm[0]), float(point_cm[1])
        poly = np.asarray(self.polygon_cm(), dtype=float)
        starts = poly
        ends = np.roll(poly, -1, axis=0)
        sx, sy = starts[:, 0], starts[:, 1]
        ex, ey = ends[:, 0], ends[:, 1]

        # Even-odd ray cast for strictly-inside.
        straddles = (sy > py) != (ey > py)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_at_y = sx + (py - sy) / (ey - sy) * (ex - sx)
        inside = bool(np.count_nonzero(straddles & (px < x_at_y)) % 2 == 1)
        if inside or margin_cm <= 0.0:
            return inside

        # Outside: accept if within margin of any edge segment.
        edge = ends - starts
        edge_len_sq = (edge * edge).sum(axis=1)
        edge_len_sq = np.where(edge_len_sq > 0.0, edge_len_sq, 1.0)
        t = np.clip(((px - sx) * edge[:, 0] + (py - sy) * edge[:, 1]) / edge_len_sq, 0.0, 1.0)
        proj_x = sx + t * edge[:, 0]
        proj_y = sy + t * edge[:, 1]
        dist_sq = (px - proj_x) ** 2 + (py - proj_y) ** 2
        return bool(dist_sq.min() <= margin_cm * margin_cm)

    @classmethod
    def from_tip_and_armpit(
        cls,
        tip_corner_cm: tuple[float, float],
        armpit_corner_cm: tuple[float, float],
        length_to_width_ratio: float = 20.0 / 3.0,
    ) -> "RedCrossSpec":
        """Build a cross from one arm's outer tip corner and its inner armpit corner.

        The two corners lie on the same arm side-edge, so ``tip - armpit`` is the
        arm axis and its length equals one side edge, ``(length - width) / 2``.
        The cross is sized *relative to that clicked distance* (preserving the
        plus proportion ``length_to_width_ratio``) rather than fixed centimetres,
        so it scales to whatever cross you click.  The center is the tip corner
        stepped back a half-length along the axis and a half-width across it.
        """
        tx, ty = tip_corner_cm
        ax, ay = armpit_corner_cm
        dx, dy = tx - ax, ty - ay
        edge = math.hypot(dx, dy)
        angle = math.atan2(dy, dx) if edge > 1e-6 else 0.0

        # edge = (length - width) / 2  and  length = ratio * width
        ratio = max(1.0 + 1e-6, float(length_to_width_ratio))
        arm_width_cm = 2.0 * edge / (ratio - 1.0)
        length_cm = ratio * arm_width_cm

        half_len = length_cm / 2.0
        half_w = arm_width_cm / 2.0
        ex, ey = math.cos(angle), math.sin(angle)   # outward arm axis
        nx, ny = -math.sin(angle), math.cos(angle)   # +90 deg perpendicular
        center = (
            tx - half_len * ex - half_w * nx,
            ty - half_len * ey - half_w * ny,
        )
        return cls(center_cm=center, angle_rad=angle, length_cm=length_cm, arm_width_cm=arm_width_cm)

    def to_dict(self) -> dict[str, Any]:
        return {
            "center_cm": [float(self.center_cm[0]), float(self.center_cm[1])],
            "angle_rad": float(self.angle_rad),
            "angle_deg": math.degrees(float(self.angle_rad)),
            "length_cm": float(self.length_cm),
            "arm_width_cm": float(self.arm_width_cm),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RedCrossSpec | None":
        try:
            center = data["center_cm"]
            return cls(
                center_cm=(float(center[0]), float(center[1])),
                angle_rad=float(data["angle_rad"]),
                length_cm=float(data.get("length_cm", 20.0)),
                arm_width_cm=float(data.get("arm_width_cm", 3.0)),
            )
        except (KeyError, TypeError, ValueError, IndexError):
            return None


@dataclass(frozen=True)
class HSVRange:
    """Single HSV threshold range."""

    lower: np.ndarray
    upper: np.ndarray


@dataclass(frozen=True)
class SmoothedBallCoordinate:
    """Smoothed field coordinate paired with the detection that produced it."""

    track_id: int
    label: str
    center_px: tuple[int, int]
    corrected_center_px: tuple[int, int]
    radius_px: int
    cm_x: float
    cm_y: float


@dataclass
class SmoothedCoordinateTrack:
    """Persistent smoothing state for one detected ball."""

    label: str
    x_cm: float
    y_cm: float
    history_x_cm: deque[float] = field(default_factory=lambda: deque(maxlen=5))
    history_y_cm: deque[float] = field(default_factory=lambda: deque(maxlen=5))
    stationary_frames: int = 0
    missed_frames: int = 0
