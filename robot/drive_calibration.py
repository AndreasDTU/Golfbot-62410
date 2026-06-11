"""Drive calibration helpers shared by the laptop app and EV3 server.

This module intentionally stays compatible with the EV3 Python runtime
currently used by ``robot/robot_server.py``.  Keep it free of dataclasses,
variable annotations, ``X | None`` types, and f-strings.
"""

import json
import math
from pathlib import Path

DEFAULT_AXLE_TRACK_MM = 252.5986772
DEFAULT_MM_PER_UNIT = 9.9664


class DriveCalibrationValues(object):
    """Encoder conversion values used by the EV3 drive server."""

    __slots__ = ("axle_track_mm", "mm_per_unit")

    def __init__(self, axle_track_mm=DEFAULT_AXLE_TRACK_MM, mm_per_unit=DEFAULT_MM_PER_UNIT):
        self.axle_track_mm = float(axle_track_mm)
        self.mm_per_unit = float(mm_per_unit)

    def __eq__(self, other):
        if not isinstance(other, DriveCalibrationValues):
            return False
        return (
            self.axle_track_mm == other.axle_track_mm
            and self.mm_per_unit == other.mm_per_unit
        )

    def __repr__(self):
        return "DriveCalibrationValues(axle_track_mm={!r}, mm_per_unit={!r})".format(
            self.axle_track_mm,
            self.mm_per_unit,
        )

    def as_dict(self):
        return {
            "axle_track_mm": self.axle_track_mm,
            "mm_per_unit": self.mm_per_unit,
        }


def is_valid_calibration_value(value):
    """Return true for finite positive calibration constants."""
    return math.isfinite(float(value)) and float(value) > 0.0


def validate_drive_calibration(values):
    """Raise ``ValueError`` when calibration values are unsafe to apply."""
    if not is_valid_calibration_value(values.axle_track_mm):
        raise ValueError("axle_track_mm must be finite and positive")
    if not is_valid_calibration_value(values.mm_per_unit):
        raise ValueError("mm_per_unit must be finite and positive")
    return values


def load_drive_calibration(path, defaults=None):
    """Load calibration JSON, returning defaults when the file is absent or invalid."""
    fallback = defaults or DriveCalibrationValues()
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        values = DriveCalibrationValues(
            axle_track_mm=float(payload["axle_track_mm"]),
            mm_per_unit=float(payload["mm_per_unit"]),
        )
        return validate_drive_calibration(values)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return fallback


def save_drive_calibration(path, values):
    """Persist validated drive calibration values as deterministic JSON."""
    validated = validate_drive_calibration(values)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(validated.as_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def normalize_angle(angle_rad):
    """Normalize an angle to ``[-pi, pi)``."""
    return (float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi


def unwrapped_heading_delta(previous_rad, current_rad):
    """Return the shortest signed heading delta between consecutive observations."""
    return normalize_angle(float(current_rad) - float(previous_rad))


def suggest_axle_track_mm(old_axle_track_mm, expected_turn_deg, actual_turn_deg):
    """Return the axle-track constant that should make future turns match expectation."""
    if not is_valid_calibration_value(old_axle_track_mm):
        raise ValueError("old_axle_track_mm must be finite and positive")
    if not is_valid_calibration_value(expected_turn_deg):
        raise ValueError("expected_turn_deg must be finite and positive")
    if not is_valid_calibration_value(actual_turn_deg):
        raise ValueError("actual_turn_deg must be finite and positive")
    return float(old_axle_track_mm) * float(expected_turn_deg) / float(actual_turn_deg)


def suggest_mm_per_unit(old_mm_per_unit, expected_distance_cm, actual_distance_cm):
    """Return the linear-distance constant that should make future moves match expectation."""
    if not is_valid_calibration_value(old_mm_per_unit):
        raise ValueError("old_mm_per_unit must be finite and positive")
    if not is_valid_calibration_value(expected_distance_cm):
        raise ValueError("expected_distance_cm must be finite and positive")
    if not is_valid_calibration_value(actual_distance_cm):
        raise ValueError("actual_distance_cm must be finite and positive")
    return float(old_mm_per_unit) * float(expected_distance_cm) / float(actual_distance_cm)


def projected_motion_cm(start_xy_cm, end_xy_cm, start_heading_rad):
    """Return forward and lateral displacement in the robot frame at measurement start."""
    dx = float(end_xy_cm[0]) - float(start_xy_cm[0])
    dy = float(end_xy_cm[1]) - float(start_xy_cm[1])
    heading = float(start_heading_rad)
    forward_cm = dx * math.cos(heading) + dy * math.sin(heading)
    lateral_cm = dx * math.sin(heading) - dy * math.cos(heading)
    return forward_cm, lateral_cm


def format_drive_calibration_response(values):
    """Return the EV3 TCP response format for drive calibration values."""
    validated = validate_drive_calibration(values)
    return "ok: drivecal axle_track_mm {:.6f} mm_per_unit {:.6f}".format(
        validated.axle_track_mm,
        validated.mm_per_unit,
    )


def parse_drive_calibration_response(response):
    """Parse the EV3 ``drivecal`` response into structured values."""
    parts = response.strip().split()
    if len(parts) != 6 or parts[0].lower() != "ok:" or parts[1].lower() != "drivecal":
        raise ValueError("unexpected drive calibration response: {!r}".format(response))
    if parts[2].lower() != "axle_track_mm" or parts[4].lower() != "mm_per_unit":
        raise ValueError("unexpected drive calibration response: {!r}".format(response))
    return validate_drive_calibration(
        DriveCalibrationValues(
            axle_track_mm=float(parts[3]),
            mm_per_unit=float(parts[5]),
        )
    )
