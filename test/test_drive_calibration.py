import math
import tempfile
import unittest
from pathlib import Path

from robot.drive_calibration import (
    DriveCalibrationValues,
    format_drive_calibration_response,
    load_drive_calibration,
    marker_offset_with_preserved_alpha,
    parse_drive_calibration_response,
    projected_motion_cm,
    save_drive_calibration,
    suggest_axle_track_mm,
    suggest_mm_per_unit,
    unwrapped_heading_delta,
)


class DriveCalibrationMathTests(unittest.TestCase):
    def test_unwrapped_heading_delta_handles_angle_wrap(self) -> None:
        delta = unwrapped_heading_delta(math.radians(170.0), math.radians(-170.0))

        self.assertAlmostEqual(math.degrees(delta), 20.0)

    def test_correction_formulas_scale_expected_over_actual(self) -> None:
        self.assertAlmostEqual(suggest_axle_track_mm(250.0, 360.0, 400.0), 225.0)
        self.assertAlmostEqual(suggest_mm_per_unit(10.0, 10.0, 8.0), 12.5)

    def test_projected_motion_reports_forward_and_lateral_components(self) -> None:
        forward, lateral = projected_motion_cm((10.0, 10.0), (10.0, 20.0), math.radians(90.0))

        self.assertAlmostEqual(forward, 10.0)
        self.assertAlmostEqual(lateral, 0.0)

    def test_marker_offset_preserves_existing_alpha(self) -> None:
        dx, dy = marker_offset_with_preserved_alpha(
            marker_xy=(110.0, 100.0),
            origin_xy=(100.0, 100.0),
            marker_yaw_rad=math.radians(90.0),
            alpha_rad=0.0,
        )

        self.assertAlmostEqual(dx, 0.0, places=6)
        self.assertAlmostEqual(dy, 10.0, places=6)

    def test_drive_calibration_json_round_trips_valid_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "robot_drive_calibration.json"
            values = DriveCalibrationValues(axle_track_mm=300.0, mm_per_unit=12.5)

            save_drive_calibration(path, values)

            self.assertEqual(load_drive_calibration(path), values)

    def test_drivecal_response_round_trips(self) -> None:
        values = DriveCalibrationValues(axle_track_mm=300.0, mm_per_unit=12.5)

        parsed = parse_drive_calibration_response(format_drive_calibration_response(values))

        self.assertEqual(parsed, values)


if __name__ == "__main__":
    unittest.main()
