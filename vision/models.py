"""Shared data models for top-down perception.

These classes intentionally carry no frame-loop orchestration.  They are kept
small so the legacy detector can import them without changing behavior while
the larger refactor proceeds in controlled steps.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum

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
