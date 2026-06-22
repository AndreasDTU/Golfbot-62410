"""Red-cross collision tracking.

Two cohesive, GUI-free pieces for recovering the manually placed cross after the
robot drives through it:

* ``CrossCollisionTracker`` — a fire-on-exit state machine that decides *when*
  to look (pure boolean logic, no perception or app dependencies).
* ``relocalize_cross`` — a pure function that re-detects *where* the cross ended
  up from a top-down frame and the prior spec.

The orchestration that wires these to the live robot pose, occupancy grid, and
route planner stays in the GUI controller.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import cv2
import numpy as np

from perception.vision.models import RedCrossSpec

if TYPE_CHECKING:
    from perception.vision.detection import RedZoneDetector
    from perception.vision.geometry import CoordinateMapper


class CrossAction(Enum):
    """Outcome of advancing the collision tracker one frame."""

    NONE = "none"
    RELOCALIZE = "relocalize"


@dataclass
class CrossCollisionTracker:
    """Fire-on-exit state machine for robot-vs-cross collisions.

    Armed while the robot sits inside the cross buffer (where it occludes the
    cross from the overhead camera), and emits a single ``RELOCALIZE`` once the
    robot has cleared the zone so the cross is visible again.  ``request_replan``
    / ``take_pending_replan`` carry the one-frame deferral that lets the
    occupancy grid rebuild with the new cross before the route is replanned.
    """

    _armed: bool = False
    _replan_pending: bool = False

    def reset(self) -> None:
        """Clear all state (e.g. when leaving AUTO mode)."""
        self._armed = False
        self._replan_pending = False

    def step(self, robot_in_buffer: bool) -> CrossAction:
        """Advance one frame; returns ``RELOCALIZE`` on the buffer-exit edge."""
        if robot_in_buffer:
            self._armed = True
            return CrossAction.NONE
        if self._armed:
            self._armed = False
            return CrossAction.RELOCALIZE
        return CrossAction.NONE

    def request_replan(self) -> None:
        """Mark that a replan is owed once the grid has rebuilt."""
        self._replan_pending = True

    def take_pending_replan(self) -> bool:
        """Return whether a replan is owed, consuming the flag."""
        pending = self._replan_pending
        self._replan_pending = False
        return pending


def relocalize_cross(
    topdown_bgr: np.ndarray,
    prior: RedCrossSpec,
    params: dict,
    mapper: "CoordinateMapper",
    detector: "RedZoneDetector",
) -> RedCrossSpec | None:
    """Re-detect the displaced cross near its prior pose.

    Returns an updated spec carrying a new ``center_cm`` (orientation and the
    fixed arm dimensions are retained), or ``None`` if no red zone is found
    within reach of the prior position.

    Only the center is re-estimated: a symmetric plus has 90-degree rotational
    symmetry, which makes bounding-box / second-moment angle estimates
    degenerate, and a robot bump translates the cross far more reliably than it
    rotates it.
    """
    camera_center_pixels = (
        float(params["camera_center_x"]),
        float(params["camera_center_y"]),
    )
    zones, _mask = detector.detect(topdown_bgr, params, camera_center_pixels)
    if not zones:
        return None

    # Gate: only accept a red zone whose center is within reach of where the
    # cross was — its extent plus a generous push allowance.  This also rejects
    # the field border / goal red, which sits far from the cross.
    robot_w = float(params.get("robot_width_cm", 20.0))
    gate_cm = prior.half_size_cm + 2.0 * robot_w
    best_zone = None
    best_dist = gate_cm
    for zone in zones:
        zc = mapper.topdown_px_to_field_cm(zone.corrected_center)
        dist = math.hypot(zc[0] - prior.center_cm[0], zc[1] - prior.center_cm[1])
        if dist <= best_dist:
            best_dist = dist
            best_zone = zone
    if best_zone is None:
        return None

    center_cm = _contour_centroid_cm(best_zone.corrected_contour, mapper)
    if center_cm is None:
        return None
    return RedCrossSpec(
        center_cm=center_cm,
        angle_rad=prior.angle_rad,
        length_cm=prior.length_cm,
        arm_width_cm=prior.arm_width_cm,
    )


def _contour_centroid_cm(
    contour_px: np.ndarray, mapper: "CoordinateMapper"
) -> tuple[float, float] | None:
    """Polygon centroid of a top-down-pixel contour, expressed in field cm.

    The contour is converted to cm before taking the centroid because the
    top-down warp flips Y and is mildly anisotropic.
    """
    pts_cm = np.array(
        [mapper.topdown_px_to_field_cm((float(p[0][0]), float(p[0][1]))) for p in contour_px],
        dtype=np.float32,
    )
    if len(pts_cm) < 3:
        return None
    moments = cv2.moments(pts_cm)
    if moments["m00"] <= 1e-6:
        cx, cy = pts_cm.mean(axis=0)
        return float(cx), float(cy)
    return float(moments["m10"] / moments["m00"]), float(moments["m01"] / moments["m00"])
