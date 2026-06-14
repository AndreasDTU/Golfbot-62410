import csv
import io
import os
import tempfile
import unittest
from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import patch

from robot.telemetry import DriveTelemetryRecorder, TelemetryFrame, log_event


def _make_dummy_objects(
    robot_pose=True,
    tracking_error=True,
):
    """Build minimal stand-ins for record_frame arguments."""
    pose = SimpleNamespace(
        x_cm=10.0, y_cm=20.0, heading_rad=0.5, tube_x_cm=11.0, tube_y_cm=21.0
    ) if robot_pose else None

    error = SimpleNamespace(
        xte_cm=1.5, signed_xte_cm=-1.5, heading_error_rad=0.1,
        segment_index=2, segment_heading_rad=0.3,
    ) if tracking_error else None

    wc = SimpleNamespace(
        last_distance_to_goal_cm=30.0,
        last_forward_scale=0.8,
        last_desired_base_speed=25.0,
        last_base_speed=24.0,
        last_edge_gain=1.0,
        last_heading_derivative=0.01,
        last_cross_track_derivative=0.02,
        last_turn_speed=3.5,
        last_heading_p_term=2.0,
        last_heading_d_term=0.5,
        last_xte_p_term=0.8,
        last_xte_d_term=0.2,
    )

    drive_runtime = SimpleNamespace(
        last_error=error,
        state=SimpleNamespace(value="TRACKING"),
        last_message="ok",
        last_command=SimpleNamespace(left_pct=20.0, right_pct=22.0),
        route_progress_segment_index=3,
    )

    return pose, drive_runtime, wc


class TestRingbufferCapacity(unittest.TestCase):
    def test_ringbuffer_capacity(self):
        maxlen = 10
        recorder = DriveTelemetryRecorder(maxlen=maxlen)
        pose, dr, wc = _make_dummy_objects()
        for _ in range(maxlen + 5):
            recorder.record_frame(pose, dr, wc, frame_dt_s=0.033)
        self.assertEqual(len(recorder._buffer), maxlen)


class TestCsvDump(unittest.TestCase):
    def test_csv_dump_creates_file_with_correct_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = DriveTelemetryRecorder(maxlen=50, dump_dir=tmpdir)
            pose, dr, wc = _make_dummy_objects()
            for _ in range(5):
                recorder.record_frame(pose, dr, wc, frame_dt_s=0.033)
            path = recorder.dump_csv(reason="test")
            self.assertIsNotNone(path)
            self.assertTrue(os.path.isfile(path))
            with open(path) as fh:
                reader = csv.DictReader(fh)
                rows = list(reader)
            self.assertEqual(len(rows), 5)
            expected_fields = {f.name for f in fields(TelemetryFrame)}
            self.assertEqual(set(reader.fieldnames), expected_fields)

    def test_csv_dump_empty_buffer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = DriveTelemetryRecorder(dump_dir=tmpdir)
            result = recorder.dump_csv(reason="empty")
            self.assertIsNone(result)


class TestRecordFrameWithNoPose(unittest.TestCase):
    def test_record_frame_with_no_pose(self):
        recorder = DriveTelemetryRecorder(maxlen=10)
        _, dr, wc = _make_dummy_objects(robot_pose=False, tracking_error=False)
        dr.last_error = None
        recorder.record_frame(None, dr, wc, frame_dt_s=0.033)
        frame = recorder._buffer[0]
        self.assertIsNone(frame.robot_x_cm)
        self.assertIsNone(frame.robot_y_cm)
        self.assertIsNone(frame.robot_heading_rad)
        self.assertIsNone(frame.xte_cm)
        self.assertIsNone(frame.signed_xte_cm)


class TestLogEventOutputFormat(unittest.TestCase):
    def test_log_event_output_format(self):
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            log_event("ROUTE", "installed", segments=12, transit=8)
        line = buf.getvalue().strip()
        self.assertIn("[ROUTE]", line)
        self.assertIn("installed", line)
        self.assertIn("segments=12", line)
        self.assertIn("transit=8", line)
        # Verify timestamp prefix format
        self.assertTrue(line.startswith("["))


if __name__ == "__main__":
    unittest.main()
