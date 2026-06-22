"""Shared test fixtures and helpers."""

from __future__ import annotations

import json
from pathlib import Path

from localization.models import RobotGeometry


def load_test_geometry() -> RobotGeometry:
    """Load RobotGeometry from the calibration file (single source of truth)."""
    path = Path(__file__).resolve().parent.parent / "data" / "robot_calibration.json"
    with path.open() as f:
        cal = json.load(f)
    geo = cal["geometry"]
    return RobotGeometry(
        width_cm=geo["width_cm"],
        front_cm=geo["front_cm"],
        rear_cm=geo["rear_cm"],
        tube_forward_cm=geo["tube_forward_cm"],
        tube_right_cm=geo["tube_right_cm"],
        tube_width_cm=geo.get("tube_width_cm", 6.0),
        mouth_radius_cm=geo.get("mouth_radius_cm", 2.0),
        unload_extension_cm=geo.get("unload_extension_cm", 30.0),
        pipe_diameter_cm=geo.get("pipe_diameter_cm", 4.5),
    )
