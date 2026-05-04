import unittest
from pathlib import Path

import numpy as np

from tools.topdown_object_detector import IdentityPreprocessor, RuntimeState, TopdownDetectorApp
from vision.models import CalibrationState


class TopdownDetectorAppShellTests(unittest.TestCase):
    def test_app_constructs_domain_services(self) -> None:
        app = TopdownDetectorApp()

        self.assertEqual(app.config.field.width_cm, 167.0)
        self.assertIsNotNone(app.mapper)
        self.assertIsNotNone(app.renderer)
        self.assertIsNotNone(app.route_facade)
        self.assertIsNotNone(app.robot_estimator)
        self.assertIsNotNone(app.robot_calibration_collector)
        self.assertIsNotNone(app.drive_guard)

    def test_identity_preprocessor_treats_frame_as_topdown(self) -> None:
        frame = np.arange(4 * 5 * 3, dtype=np.uint8).reshape((4, 5, 3))
        result = IdentityPreprocessor().process(frame, use_aruco=False)

        np.testing.assert_array_equal(result.undistorted, frame)
        np.testing.assert_array_equal(result.topdown, frame)
        np.testing.assert_array_equal(result.normalized, frame)
        self.assertEqual(result.calibration_state, CalibrationState.CALIBRATED_MANUAL)
        np.testing.assert_allclose(result.transform_matrix, np.eye(3, dtype=np.float32))

    def test_runtime_state_clear_route(self) -> None:
        state = RuntimeState()
        state.route_plan.points.append(object())

        state.clear_route()

        self.assertEqual(state.route_plan.points, [])
        self.assertIsNone(state.route_plan.active_target)
        self.assertEqual(state.route_plan.pickup_poses, [])
        self.assertIsNone(state.route_plan.unload_pose)
        self.assertIsNone(state.route_plan.unload_goal_cm)
        self.assertIsNone(state.route_failed_ball_signature)
        self.assertIsNone(state.route_failed_unload_extension_cm)
        self.assertIsNone(state.route_failed_start_signature)

    def test_parse_args_defaults(self) -> None:
        original_argv = __import__("sys").argv
        try:
            __import__("sys").argv = ["topdown_object_detector.py"]
            args = TopdownDetectorApp.parse_args()
        finally:
            __import__("sys").argv = original_argv

        self.assertFalse(args.live)
        self.assertEqual(args.image, Path(__file__).resolve().parents[1] / "Bane_undistorted_transformed_close_balls.png")
        self.assertIsNone(args.video)
        self.assertFalse(args.drive)


if __name__ == "__main__":
    unittest.main()
