import math
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from pathfinding.planner import HybridAStarPlanner, LegacyAStarPlanner, RobotFootprintCollisionChecker, RoutePlanningFacade
from robot.control import DriveSafetyGuard, WheelCommandController
from robot.io import UdpWheelDispatcher
from robot.localization import RobotCalibrationCollector, RobotMarkerDetector, RobotPoseEstimator
from tools import topdown_object_detector as detector
from vision.calibration import HomographyCalibrator, UndistortionProvider
from vision.config import AppConfig
from vision.detection import RedZoneDetector, YoloBallDetector
from vision.debug import DebugRenderer
from vision.geometry import CoordinateMapper, ParallaxCorrector
from vision.grid_mapping import OccupancyGridBuilder
from vision.models import ParallaxConfig
from vision.pipeline import VisionPipeline
from vision.preprocessing import FramePreprocessor
from vision.tracking import BallCoordinateSmoother


class TopdownDetectorCharacterizationTests(unittest.TestCase):
    def test_order_points_returns_stable_clockwise_corner_order(self) -> None:
        unordered = np.array(
            [
                [9.0, 7.0],
                [1.0, 1.0],
                [9.0, 1.0],
                [1.0, 7.0],
            ],
            dtype=np.float32,
        )

        ordered = detector.order_points(unordered)

        np.testing.assert_allclose(
            ordered,
            np.array(
                [
                    [1.0, 1.0],
                    [9.0, 1.0],
                    [9.0, 7.0],
                    [1.0, 7.0],
                ],
                dtype=np.float32,
            ),
        )

    def test_topdown_field_coordinate_round_trip_preserves_known_points(self) -> None:
        points_cm = [
            (0.0, 0.0),
            (detector.FIELD_WIDTH_CM, detector.FIELD_HEIGHT_CM),
            (83.5, 60.75),
        ]

        for point_cm in points_cm:
            pixel = detector.field_cm_to_topdown_pixel(point_cm)
            round_trip_cm = detector.topdown_px_to_field_cm(pixel)

            self.assertAlmostEqual(round_trip_cm[0], point_cm[0], places=6)
            self.assertAlmostEqual(round_trip_cm[1], point_cm[1], places=6)

    def test_parallax_correction_keeps_input_when_camera_plane_is_degenerate(self) -> None:
        corrected = detector.correct_parallax(
            pixel_coord=(123, 456),
            z_object_cm=2.0,
            h_cam_cm=7.0,
            z_calib_cm=7.0,
            camera_center_pixels=(100.0, 400.0),
        )

        self.assertEqual(corrected, (123, 456))

    def test_parallax_correction_uses_existing_radial_projection_math(self) -> None:
        corrected = detector.correct_parallax(
            pixel_coord=(120, 90),
            z_object_cm=2.0,
            h_cam_cm=10.0,
            z_calib_cm=0.0,
            camera_center_pixels=(100.0, 100.0),
        )

        self.assertEqual(corrected, (116, 92))

    def test_occupancy_grid_from_red_zone_is_deterministic(self) -> None:
        contour = np.array(
            [
                [[100, 100]],
                [[140, 100]],
                [[140, 140]],
                [[100, 140]],
            ],
            dtype=np.int32,
        )
        zone = detector.RedZoneDetection(
            contour=contour,
            corrected_contour=contour.copy(),
            bounding_box=(100, 100, 40, 40),
            center=(120, 120),
            corrected_center=(120, 120),
            area=1600.0,
        )

        grid_1 = detector.build_occupancy_grid((600, 800, 3), [zone], dilate_for_legacy=False)
        grid_2 = detector.build_occupancy_grid((600, 800, 3), [zone], dilate_for_legacy=False)

        self.assertEqual(grid_1.shape, (detector.FIELD_GRID_HEIGHT_CM, detector.FIELD_GRID_WIDTH_CM))
        self.assertGreater(int(grid_1.sum()), 0)
        np.testing.assert_array_equal(grid_1, grid_2)

    def test_hybrid_a_star_simple_route_is_deterministic(self) -> None:
        grid = np.zeros((detector.FIELD_GRID_HEIGHT_CM, detector.FIELD_GRID_WIDTH_CM), dtype=np.uint8)
        start_pose = detector.HybridPose(x_cm=20.0, y_cm=20.0, theta_rad=0.0)
        goal_node = detector.field_metric_cm_to_grid_node((36.0, 20.0))
        geometry = detector.robot_geometry_from_params(None)
        config = detector.HybridPlannerConfig(max_expansions=500)

        route_1 = detector.hybrid_a_star_search(grid, start_pose, goal_node, geometry, config)
        route_2 = detector.hybrid_a_star_search(grid, start_pose, goal_node, geometry, config)

        self.assertGreaterEqual(len(route_1), 2)
        self.assertEqual(route_1, route_2)
        self.assertTrue(all(math.isfinite(pose.x_cm) and math.isfinite(pose.y_cm) for pose in route_1))

    def test_extracted_model_defaults_match_legacy_constants(self) -> None:
        config = detector.HybridPlannerConfig()
        self.assertEqual(config.step_cm, detector.HYBRID_STEP_CM)
        self.assertEqual(config.theta_bins, detector.HYBRID_THETA_BINS)
        self.assertEqual(config.goal_tolerance_cm, detector.HYBRID_GOAL_TOLERANCE_CM)
        self.assertEqual(config.max_expansions, detector.HYBRID_MAX_EXPANSIONS)
        self.assertEqual(config.translation_directions, detector.HYBRID_TRANSLATION_DIRECTIONS)
        self.assertEqual(config.rotation_deltas_rad, detector.HYBRID_ROTATION_DELTAS_RAD)
        self.assertEqual(config.in_place_rotation_cost, detector.HYBRID_IN_PLACE_ROTATION_COST)

        runtime = detector.RobotCalibrationRuntime()
        self.assertEqual(tuple(runtime.collected_points), detector.ROBOT_MARKER_IDS)

    def test_extracted_config_defaults_match_legacy_constants(self) -> None:
        config = AppConfig.from_repo_root(detector.REPO_ROOT)

        self.assertEqual(config.field.width_cm, detector.FIELD_WIDTH_CM)
        self.assertEqual(config.field.height_cm, detector.FIELD_HEIGHT_CM)
        self.assertEqual(config.field.grid_width_cm, detector.FIELD_GRID_WIDTH_CM)
        self.assertEqual(config.field.grid_height_cm, detector.FIELD_GRID_HEIGHT_CM)
        self.assertEqual(config.field.ball_height_cm, detector.Z_BALL_CM)
        self.assertEqual(config.field.floor_height_cm, detector.Z_FLOOR_CM)

        self.assertEqual(config.paths.calibration_file, detector.CALIBRATION_FILE)
        self.assertEqual(config.paths.robot_calibration_file, detector.ROBOT_CALIBRATION_FILE)
        self.assertEqual(config.paths.default_image, detector.DEFAULT_IMAGE)
        self.assertEqual(config.paths.default_video_dir, detector.DEFAULT_VIDEO_DIR)

        self.assertEqual(config.camera.use_live_feed, detector.USE_LIVE_FEED)
        self.assertEqual(config.camera.camera_index, detector.CAMERA_INDEX)
        self.assertEqual(config.camera.topdown_warp_size, detector.TOPDOWN_WARP_SIZE)
        self.assertEqual(config.camera.required_aruco_ids, detector.REQUIRED_ARUCO_IDS)
        self.assertEqual(config.camera.wall_thickness_cm, detector.WALL_THICKNESS_CM)
        self.assertEqual(config.camera.marker_outer_offset_cm, detector.MARKER_OUTER_OFFSET_CM)

        self.assertEqual(config.windows.main_window_name, detector.WINDOW_NAME)
        self.assertEqual(config.windows.mask_window_name, detector.MASK_WINDOW_NAME)
        self.assertEqual(config.windows.schematic_window_name, detector.SCHEMATIC_WINDOW_NAME)
        self.assertEqual(config.windows.schematic_width_px, detector.SCHEMATIC_WIDTH_PX)
        self.assertEqual(config.windows.schematic_height_px, detector.SCHEMATIC_HEIGHT_PX)
        self.assertEqual(config.windows.control_window_size, detector.CONTROL_WINDOW_SIZE)

        self.assertEqual(config.robot.marker_ids, detector.ROBOT_MARKER_IDS)
        self.assertEqual(config.robot.marker_height_cm, detector.ROBOT_MARKER_HEIGHT_CM)
        self.assertEqual(config.robot.tuned_footprint_width_cm, detector.ROBOT_TUNED_FOOTPRINT_WIDTH_CM)
        self.assertEqual(config.robot.tuned_footprint_front_from_origin_cm, detector.ROBOT_TUNED_FOOTPRINT_FRONT_FROM_ORIGIN_CM)
        self.assertEqual(config.robot.tuned_footprint_rear_from_origin_cm, detector.ROBOT_TUNED_FOOTPRINT_REAR_FROM_ORIGIN_CM)
        self.assertEqual(config.robot.tuned_tube_offset_cm, detector.ROBOT_TUNED_TUBE_OFFSET_CM)
        self.assertEqual(config.robot.tuned_tube_right_offset_cm, detector.ROBOT_TUNED_TUBE_RIGHT_OFFSET_CM)

        self.assertEqual(config.trackbars.names, detector.TRACKBAR_NAMES)
        self.assertEqual(config.trackbars.windows(config.windows), detector.TRACKBAR_WINDOWS)
        self.assertEqual(config.planner.theta_bins, detector.HYBRID_THETA_BINS)
        self.assertEqual(config.planner.step_cm, detector.HYBRID_STEP_CM)
        self.assertEqual(config.planner.route_target_reached_cm, detector.ROUTE_TARGET_REACHED_CM)
        self.assertEqual(config.drive.robot_ip, detector.ROBOT_IP)
        self.assertEqual(config.drive.robot_udp_port, detector.ROBOT_UDP_PORT)
        self.assertEqual(config.drive.robot_command_format, detector.ROBOT_COMMAND_FORMAT)
        self.assertEqual(config.robot_calibration.min_robot_spin_points, detector.MIN_ROBOT_SPIN_POINTS)
        self.assertEqual(config.robot_calibration.ellipse_warning_ratio, detector.ELLIPSE_WARNING_RATIO)

    def test_coordinate_mapper_matches_legacy_coordinate_functions(self) -> None:
        mapper = CoordinateMapper()
        point_px = np.array([321.5, 222.25], dtype=np.float32)
        point_cm = (42.75, 88.5)
        source_size = (800, 600)

        self.assertEqual(mapper.topdown_px_to_field_cm(point_px), detector.topdown_px_to_field_cm(point_px))
        self.assertEqual(mapper.field_cm_to_topdown_pixel(point_cm), detector.field_cm_to_topdown_pixel(point_cm))
        self.assertEqual(mapper.pixel_to_field_cm((321, 222), source_size), detector.pixel_to_field_cm((321, 222), source_size))
        self.assertEqual(
            mapper.pixel_float_to_field_cm(np.array([321.2, 222.6], dtype=np.float32), source_size),
            detector.pixel_float_to_field_cm(np.array([321.2, 222.6], dtype=np.float32), source_size),
        )
        self.assertEqual(mapper.field_metric_cm_to_grid_node(point_cm), detector.field_metric_cm_to_grid_node(point_cm))
        self.assertEqual(mapper.field_metric_cm_to_schematic(point_cm), detector.field_metric_cm_to_schematic(point_cm))
        self.assertEqual(mapper.schematic_to_field_metric_cm((123, 456)), detector.schematic_to_field_metric_cm((123, 456)))
        self.assertEqual(mapper.source_point_to_field_cm((321, 222), source_size), detector.source_point_to_field_cm((321, 222), source_size))
        self.assertEqual(mapper.field_cm_to_schematic((33, 44)), detector.field_cm_to_schematic((33, 44)))
        self.assertEqual(
            mapper.map_point_between_frames((12, 34), (800, 600), (900, 600)),
            detector.map_point_between_frames((12, 34), (800, 600), (900, 600)),
        )
        self.assertEqual(
            mapper.field_cm_to_topdown_px_unflipped(12.5, 34.5),
            detector.field_cm_to_topdown_px(12.5, 34.5),
        )

    def test_coordinate_mapper_matches_legacy_array_geometry(self) -> None:
        mapper = CoordinateMapper()
        unordered = np.array([[9.0, 7.0], [1.0, 1.0], [9.0, 1.0], [1.0, 7.0]], dtype=np.float32)
        contour = np.array([[[100, 100]], [[140, 100]], [[140, 140]], [[100, 140]]], dtype=np.int32)
        camera_matrix = np.array([[100.0, 0.0, 320.0], [0.0, 100.0, 240.0], [0.0, 0.0, 1.0]])
        homography = np.eye(3, dtype=np.float64)

        np.testing.assert_allclose(mapper.order_points(unordered), detector.order_points(unordered))
        np.testing.assert_allclose(mapper.destination_corners(), detector.destination_corners(detector.TOPDOWN_WARP_SIZE))
        np.testing.assert_allclose(
            mapper.build_manual_topdown_transform(unordered),
            detector.build_manual_topdown_transform(unordered),
        )
        np.testing.assert_array_equal(
            mapper.contour_to_field_grid(contour, (800, 600)),
            detector.contour_to_field_grid(contour, (800, 600)),
        )

        projection = mapper.project_principal_point_to_topdown(camera_matrix, homography)
        legacy_projection = detector.project_principal_point_to_topdown(camera_matrix, homography)
        self.assertIsNotNone(projection)
        self.assertIsNotNone(legacy_projection)
        np.testing.assert_allclose(projection.principal_point_px, legacy_projection.principal_point_px)
        np.testing.assert_allclose(projection.camera_center_px, legacy_projection.camera_center_px)

    def test_parallax_corrector_matches_legacy_parallax_functions(self) -> None:
        corrector = ParallaxCorrector()
        contour = np.array([[[100, 100]], [[140, 100]], [[140, 140]], [[100, 140]]], dtype=np.int32)
        config = ParallaxConfig(
            marker_height_cm=9.0,
            camera_height_cm=179.0,
            calibration_plane_height_cm=7.0,
            camera_center=np.array([400.0, 300.0], dtype=np.float32),
        )

        self.assertEqual(
            corrector.correct_point((120, 90), 2.0, 10.0, 0.0, (100.0, 100.0)),
            detector.correct_parallax((120, 90), 2.0, 10.0, 0.0, (100.0, 100.0)),
        )
        np.testing.assert_array_equal(
            corrector.correct_contour(contour, 0.0, 179.0, 7.0, (400.0, 300.0)),
            detector.correct_contour_parallax(contour, 0.0, 179.0, 7.0, (400.0, 300.0)),
        )
        np.testing.assert_allclose(
            corrector.correct_point_float(np.array([250.0, 150.0], dtype=np.float32), config),
            detector.parallax_correct_point_float(np.array([250.0, 150.0], dtype=np.float32), config),
        )

    def test_undistortion_provider_matches_legacy_map_loading(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as tmp_dir:
            calibration_file = Path(tmp_dir) / "calibration_data.npz"
            camera_matrix = np.array(
                [
                    [120.0, 0.0, 4.0],
                    [0.0, 120.0, 3.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            dist_coeffs = np.zeros((4, 1), dtype=np.float64)
            image_size = np.array([8, 6], dtype=np.int32)
            np.savez(calibration_file, K=camera_matrix, D=dist_coeffs, image_size=image_size)

            provider = UndistortionProvider(calibration_file, balance=0.0)
            maps = provider.maps
            legacy_camera_matrix, legacy_image_size, legacy_map1, legacy_map2 = detector.load_undistortion_maps(
                calibration_file,
                0.0,
            )

            self.assertEqual(UndistortionProvider.load_calibration_image_size(calibration_file), detector.load_calibration_image_size(calibration_file))
            self.assertEqual(maps.image_size, legacy_image_size)
            np.testing.assert_allclose(maps.undistorted_camera_matrix, legacy_camera_matrix)
            np.testing.assert_allclose(maps.map1, legacy_map1)
            np.testing.assert_allclose(maps.map2, legacy_map2)

    def test_homography_calibrator_matches_legacy_homography_helpers(self) -> None:
        calibrator = HomographyCalibrator()
        marker_centers = {
            0: np.array([10.0, 20.0], dtype=np.float32),
            1: np.array([790.0, 20.0], dtype=np.float32),
            2: np.array([790.0, 580.0], dtype=np.float32),
            3: np.array([10.0, 580.0], dtype=np.float32),
        }
        manual_points = [(10, 20), (790, 20), (790, 580), (10, 580)]
        corners = [
            np.array([[[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]]], dtype=np.float32),
            np.array([[[10.0, 0.0], [12.0, 0.0], [12.0, 2.0], [10.0, 2.0]]], dtype=np.float32),
        ]
        ids = np.array([[0], [99]], dtype=np.int32)

        np.testing.assert_allclose(calibrator.aruco_destination_points(), detector.aruco_destination_points())
        np.testing.assert_allclose(calibrator.topdown_field_corners(), detector.topdown_field_corners())
        np.testing.assert_allclose(
            calibrator.build_auto_topdown_transform(marker_centers),
            detector.build_auto_topdown_transform(marker_centers),
        )
        np.testing.assert_allclose(
            calibrator.build_manual_topdown_transform(manual_points),
            detector.build_manual_topdown_transform(manual_points),
        )
        extracted = calibrator.extract_required_marker_centers(corners, ids)
        legacy_extracted = detector.extract_required_marker_centers(corners, ids)
        self.assertEqual(set(extracted), set(legacy_extracted))
        np.testing.assert_allclose(extracted[0], legacy_extracted[0])

    def test_frame_preprocessor_matches_legacy_undistort_then_warp(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as tmp_dir:
            calibration_file = Path(tmp_dir) / "calibration_data.npz"
            camera_matrix = np.array(
                [
                    [120.0, 0.0, 4.0],
                    [0.0, 120.0, 3.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            dist_coeffs = np.zeros((4, 1), dtype=np.float64)
            image_size = np.array([8, 6], dtype=np.int32)
            np.savez(calibration_file, K=camera_matrix, D=dist_coeffs, image_size=image_size)

            provider = UndistortionProvider(calibration_file, balance=0.0)
            app_config = AppConfig.from_repo_root(detector.REPO_ROOT)
            camera_config = app_config.camera.__class__(topdown_warp_size=(8, 6))
            calibrator = HomographyCalibrator(
                field_config=app_config.field,
                camera_config=camera_config,
                mapper=CoordinateMapper(app_config.field, camera_config, app_config.windows),
            )
            manual_points = [(0, 0), (7, 0), (7, 5), (0, 5)]
            transform = calibrator.set_manual_points(manual_points)
            preprocessor = FramePreprocessor(provider, calibrator, camera_config)
            frame = np.arange(8 * 6 * 3, dtype=np.uint8).reshape((6, 8, 3))

            result = preprocessor.process(frame, use_aruco=False, normalize_illumination=False)
            legacy_undistorted = cv2.remap(
                frame,
                provider.maps.map1,
                provider.maps.map2,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
            legacy_topdown = cv2.warpPerspective(legacy_undistorted, transform, camera_config.topdown_warp_size)

            np.testing.assert_array_equal(result.undistorted, legacy_undistorted)
            np.testing.assert_array_equal(result.topdown, legacy_topdown)
            np.testing.assert_array_equal(result.normalized, legacy_topdown)
            self.assertIsNotNone(result.camera_ground_projection)

    def test_frame_preprocessor_illumination_normalization_preserves_shape(self) -> None:
        with tempfile.TemporaryDirectory(dir="/private/tmp") as tmp_dir:
            calibration_file = Path(tmp_dir) / "calibration_data.npz"
            camera_matrix = np.array(
                [
                    [120.0, 0.0, 4.0],
                    [0.0, 120.0, 3.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            dist_coeffs = np.zeros((4, 1), dtype=np.float64)
            image_size = np.array([8, 6], dtype=np.int32)
            np.savez(calibration_file, K=camera_matrix, D=dist_coeffs, image_size=image_size)

            provider = UndistortionProvider(calibration_file, balance=0.0)
            app_config = AppConfig.from_repo_root(detector.REPO_ROOT)
            camera_config = app_config.camera.__class__(topdown_warp_size=(8, 6))
            calibrator = HomographyCalibrator(app_config.field, camera_config)
            calibrator.set_manual_points([(0, 0), (7, 0), (7, 5), (0, 5)])
            preprocessor = FramePreprocessor(provider, calibrator, camera_config, normalize_illumination=True)
            frame = np.full((6, 8, 3), 64, dtype=np.uint8)
            frame[:, :, 1] = 128

            result = preprocessor.process(frame, use_aruco=False)

            self.assertIsNotNone(result.normalized)
            self.assertEqual(result.normalized.shape, result.topdown.shape)
            self.assertEqual(result.normalized.dtype, np.uint8)

    def test_red_zone_detector_matches_legacy_red_zone_detection(self) -> None:
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        frame[30:70, 40:90] = (0, 0, 255)
        params = {
            "red_1": detector.HSVRange(
                lower=np.array([0, 100, 100], dtype=np.uint8),
                upper=np.array([12, 255, 255], dtype=np.uint8),
            ),
            "red_2": detector.HSVRange(
                lower=np.array([165, 100, 100], dtype=np.uint8),
                upper=np.array([179, 255, 255], dtype=np.uint8),
            ),
            "red_min_area": 20.0,
            "h_cam_cm": 179.0,
            "z_calib_cm": 7.0,
        }
        camera_center = (80.0, 60.0)

        new_detections, new_mask = RedZoneDetector().detect(frame, params, camera_center)
        legacy_detections, legacy_mask = detector.detect_red_zones(frame, params, camera_center)

        np.testing.assert_array_equal(new_mask, legacy_mask)
        self.assertEqual(len(new_detections), len(legacy_detections))
        self.assertEqual(new_detections[0].bounding_box, legacy_detections[0].bounding_box)
        self.assertEqual(new_detections[0].center, legacy_detections[0].center)
        self.assertEqual(new_detections[0].corrected_center, legacy_detections[0].corrected_center)
        self.assertEqual(new_detections[0].area, legacy_detections[0].area)
        np.testing.assert_array_equal(new_detections[0].corrected_contour, legacy_detections[0].corrected_contour)

    def test_yolo_ball_detector_matches_legacy_yolo_postprocessing_with_fake_model(self) -> None:
        class FakeTensor:
            def __init__(self, value):
                self.value = value

            def cpu(self):
                return self

            def tolist(self):
                return self.value

            def __float__(self):
                return float(self.value)

            def __int__(self):
                return int(self.value)

        class FakeBox:
            def __init__(self, conf, cls, xyxy):
                self.conf = [FakeTensor(conf)]
                self.cls = [FakeTensor(cls)]
                self.xyxy = [FakeTensor(xyxy)]

        class FakeResult:
            names = {0: "branca", 1: "laranja", 2: "ignored"}

            def __init__(self):
                self.boxes = [
                    FakeBox(0.9, 0, [10.2, 20.1, 30.4, 44.6]),
                    FakeBox(0.95, 1, [50.0, 60.0, 72.0, 82.0]),
                    FakeBox(0.1, 0, [1.0, 1.0, 4.0, 4.0]),
                    FakeBox(0.9, 2, [5.0, 5.0, 10.0, 10.0]),
                ]

        class FakeModel:
            def __call__(self, frame, verbose=False):
                return [FakeResult()]

        frame = np.zeros((100, 120, 3), dtype=np.uint8)
        params = {
            "yolo_confidence": 0.5,
            "yolo_min_area": 20.0,
            "yolo_max_area": 2000.0,
            "h_cam_cm": 179.0,
            "z_calib_cm": 7.0,
        }
        detector_obj = YoloBallDetector(Path("fake.pt"), model=FakeModel())
        white, orange, masks = detector_obj.detect(frame, params, (60.0, 50.0))

        self.assertEqual(len(white), 1)
        self.assertEqual(len(orange), 1)
        self.assertEqual(white[0].label, "white")
        self.assertEqual(white[0].center, (20, 32))
        self.assertEqual(orange[0].label, "orange")
        self.assertEqual(orange[0].center, (61, 71))
        self.assertGreater(int(masks["white"].sum()), 0)
        self.assertGreater(int(masks["orange"].sum()), 0)

    def test_ball_coordinate_smoother_matches_legacy_smoother(self) -> None:
        contour = np.array([[[10, 10]], [[20, 10]], [[20, 20]], [[10, 20]]], dtype=np.int32)
        frames = [
            [
                detector.BallDetection("white", (10, 10), (10, 10), 4, contour, 100.0, 0.9),
                detector.BallDetection("orange", (70, 50), (70, 50), 5, contour, 121.0, 0.8),
            ],
            [
                detector.BallDetection("white", (12, 11), (12, 11), 4, contour, 100.0, 0.9),
                detector.BallDetection("orange", (72, 51), (72, 51), 5, contour, 121.0, 0.8),
            ],
            [
                detector.BallDetection("white", (12, 11), (12, 11), 4, contour, 100.0, 0.9),
            ],
        ]

        legacy_smoother = detector.BallCoordinateSmoother()
        new_smoother = BallCoordinateSmoother()
        for detections in frames:
            legacy = legacy_smoother.update(detections, (100, 120, 3))
            new = new_smoother.update(detections, (100, 120, 3))
            self.assertEqual(legacy, new)

        self.assertEqual(set(legacy_smoother.tracks), set(new_smoother.tracks))
        self.assertEqual(legacy_smoother.next_track_id, new_smoother.next_track_id)

    def test_occupancy_grid_builder_matches_legacy_grid_mapping(self) -> None:
        contour = np.array(
            [
                [[100, 100]],
                [[140, 100]],
                [[140, 140]],
                [[100, 140]],
            ],
            dtype=np.int32,
        )
        zone = detector.RedZoneDetection(
            contour=contour,
            corrected_contour=contour.copy(),
            bounding_box=(100, 100, 40, 40),
            center=(120, 120),
            corrected_center=(120, 120),
            area=1600.0,
        )
        builder = OccupancyGridBuilder()

        np.testing.assert_array_equal(
            builder.contour_to_field_grid(contour, (800, 600)),
            detector.contour_to_field_grid(contour, (800, 600)),
        )
        np.testing.assert_array_equal(
            builder.build((600, 800, 3), [zone], dilate_for_legacy=False),
            detector.build_occupancy_grid((600, 800, 3), [zone], dilate_for_legacy=False),
        )
        np.testing.assert_array_equal(
            builder.build((600, 800, 3), [zone], dilate_for_legacy=True),
            detector.build_occupancy_grid((600, 800, 3), [zone], dilate_for_legacy=True),
        )

    def test_vision_pipeline_orchestrates_components_and_result(self) -> None:
        class FakePreprocessor:
            def process(self, frame, use_aruco=True, normalize_illumination=None):
                self.last_use_aruco = use_aruco
                self.last_normalize = normalize_illumination
                return type(
                    "FakePreprocessedFrame",
                    (),
                    {
                        "undistorted": frame,
                        "topdown": frame.copy(),
                        "normalized": frame.copy(),
                        "calibration_state": detector.CalibrationState.CALIBRATED_MANUAL,
                        "transform_matrix": np.eye(3, dtype=np.float32),
                        "camera_ground_projection": None,
                        "homography_result": None,
                    },
                )()

        class FakeRedZoneDetector:
            def detect(self, frame, params, camera_center_pixels):
                self.last_camera_center_pixels = camera_center_pixels
                contour = np.array([[[1, 1]], [[3, 1]], [[3, 3]], [[1, 3]]], dtype=np.int32)
                zone = detector.RedZoneDetection(contour, contour.copy(), (1, 1, 2, 2), (2, 2), (2, 2), 4.0)
                return [zone], np.ones(frame.shape[:2], dtype=np.uint8)

        class FakeBallDetector:
            def detect(self, frame, params, camera_center_pixels):
                contour = np.array([[[5, 5]], [[9, 5]], [[9, 9]], [[5, 9]]], dtype=np.int32)
                white = [detector.BallDetection("white", (7, 7), (7, 7), 2, contour, 16.0, 0.9)]
                masks = {
                    "white": np.ones(frame.shape[:2], dtype=np.uint8),
                    "orange": np.zeros(frame.shape[:2], dtype=np.uint8),
                }
                return white, [], masks

        class FakeSmoother:
            def update(self, detections, frame_shape):
                self.last_detections = detections
                return [
                    detector.SmoothedBallCoordinate(
                        track_id=0,
                        label=detections[0].label,
                        center_px=detections[0].center,
                        corrected_center_px=detections[0].corrected_center,
                        radius_px=detections[0].radius_px,
                        cm_x=1.0,
                        cm_y=2.0,
                    )
                ]

        class FakeGridBuilder:
            def build(self, frame_shape, red_zones, dilate_for_legacy=True):
                self.last_dilate = dilate_for_legacy
                return np.zeros((3, 4), dtype=np.uint8)

        preprocessor = FakePreprocessor()
        red_detector = FakeRedZoneDetector()
        ball_detector = FakeBallDetector()
        smoother = FakeSmoother()
        grid_builder = FakeGridBuilder()
        pipeline = VisionPipeline(
            preprocessor=preprocessor,
            red_zone_detector=red_detector,
            ball_detector=ball_detector,
            ball_smoother=smoother,
            occupancy_grid_builder=grid_builder,
            build_legacy_dilated_grid=False,
        )
        frame = np.zeros((10, 12, 3), dtype=np.uint8)
        params = {
            "red_1": detector.HSVRange(np.array([0, 0, 0], dtype=np.uint8), np.array([1, 1, 1], dtype=np.uint8)),
            "red_2": detector.HSVRange(np.array([2, 2, 2], dtype=np.uint8), np.array([3, 3, 3], dtype=np.uint8)),
            "red_min_area": 1.0,
            "yolo_confidence": 0.5,
            "yolo_min_area": 1.0,
            "yolo_max_area": 100.0,
            "h_cam_cm": 10.0,
            "z_calib_cm": 0.0,
            "camera_center_x": 6.0,
            "camera_center_y": 5.0,
        }

        result = pipeline.process(frame, params=params, use_aruco=False, normalize_illumination=False)

        self.assertEqual(result.metadata["status"], "ok")
        self.assertEqual(len(result.red_zones), 1)
        self.assertEqual(len(result.white_balls), 1)
        self.assertEqual(len(result.orange_balls), 0)
        self.assertEqual(result.all_balls, result.white_balls + result.orange_balls)
        self.assertEqual(len(result.smoothed_ball_coordinates), 1)
        self.assertEqual(result.occupancy_grid.shape, (3, 4))
        self.assertFalse(preprocessor.last_use_aruco)
        self.assertFalse(preprocessor.last_normalize)
        self.assertEqual(red_detector.last_camera_center_pixels, (6.0, 5.0))
        self.assertEqual(smoother.last_detections, result.all_balls)
        self.assertFalse(grid_builder.last_dilate)

    def test_vision_pipeline_returns_empty_result_without_topdown_transform(self) -> None:
        class FakePreprocessor:
            def process(self, frame, use_aruco=True, normalize_illumination=None):
                return type(
                    "FakePreprocessedFrame",
                    (),
                    {
                        "undistorted": frame,
                        "topdown": None,
                        "normalized": None,
                        "calibration_state": detector.CalibrationState.NEEDS_CALIBRATION,
                        "transform_matrix": None,
                        "camera_ground_projection": None,
                        "homography_result": None,
                    },
                )()

        class FailingDetector:
            def detect(self, *_args, **_kwargs):
                raise AssertionError("detectors must not run without a top-down frame")

        pipeline = VisionPipeline(
            preprocessor=FakePreprocessor(),
            red_zone_detector=FailingDetector(),
            ball_detector=FailingDetector(),
        )
        frame = np.zeros((10, 12, 3), dtype=np.uint8)

        result = pipeline.process(frame, use_aruco=False)

        self.assertEqual(result.metadata["status"], "missing_topdown_transform")
        self.assertEqual(result.red_zones, [])
        self.assertEqual(result.white_balls, [])
        self.assertEqual(result.orange_balls, [])
        self.assertEqual(result.smoothed_ball_coordinates, [])
        self.assertIsNone(result.occupancy_grid)

    def test_robot_marker_detector_observations_match_legacy_extraction(self) -> None:
        class FakeDetector:
            def detectMarkers(self, frame):
                corners = [
                    np.array([[[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0]]], dtype=np.float32),
                    np.array([[[40.0, 10.0], [50.0, 10.0], [50.0, 20.0], [40.0, 20.0]]], dtype=np.float32),
                ]
                ids = np.array([[4], [99]], dtype=np.int32)
                return corners, ids, []

        frame = np.zeros((80, 100, 3), dtype=np.uint8)
        parallax_config = detector.ParallaxConfig(
            marker_height_cm=9.0,
            camera_height_cm=179.0,
            calibration_plane_height_cm=7.0,
            camera_center=np.array([50.0, 40.0], dtype=np.float32),
        )
        marker_detector = RobotMarkerDetector(
            marker_ids=detector.ROBOT_MARKER_IDS,
            dictionary=object(),
            detector_or_parameters=FakeDetector(),
        )
        new_observations = marker_detector.extract_observations(frame, parallax_config)
        legacy_observations = detector.extract_robot_marker_observations(
            frame,
            object(),
            FakeDetector(),
            detector.ROBOT_MARKER_IDS,
            parallax_config,
        )

        self.assertEqual(set(new_observations), set(legacy_observations))
        np.testing.assert_allclose(new_observations[4].center, legacy_observations[4].center)
        np.testing.assert_allclose(new_observations[4].ground_center, legacy_observations[4].ground_center)
        np.testing.assert_allclose(new_observations[4].ground_corners, legacy_observations[4].ground_corners)
        self.assertAlmostEqual(new_observations[4].yaw_rad, legacy_observations[4].yaw_rad)

    def test_robot_pose_estimator_matches_legacy_pose_estimation(self) -> None:
        class FakeRobotMarkerDetector:
            def extract_observations(self, frame, parallax_config):
                corners = np.array(
                    [[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0]],
                    dtype=np.float32,
                )
                return {
                    4: detector.RobotMarkerObservation(
                        marker_id=4,
                        center=corners.mean(axis=0),
                        ground_center=np.array([30.0, 40.0], dtype=np.float32),
                        corners=corners,
                        ground_corners=corners,
                        yaw_rad=0.25,
                    )
                }

        def fake_extract(_frame, _dictionary, _detector_or_parameters, _marker_ids, _parallax_config):
            return FakeRobotMarkerDetector().extract_observations(_frame, _parallax_config)

        frame = np.zeros((100, 120, 3), dtype=np.uint8)
        params = {
            "h_cam_cm": 179.0,
            "z_calib_cm": 7.0,
            "camera_center_x": 60.0,
            "camera_center_y": 50.0,
            "heading_tuning_rad": 0.1,
            "robot_width_cm": detector.ROBOT_TUNED_FOOTPRINT_WIDTH_CM,
            "robot_front_cm": detector.ROBOT_TUNED_FOOTPRINT_FRONT_FROM_ORIGIN_CM,
            "robot_rear_cm": detector.ROBOT_TUNED_FOOTPRINT_REAR_FROM_ORIGIN_CM,
            "tube_forward_cm": detector.ROBOT_TUNED_TUBE_OFFSET_CM,
            "tube_right_cm": detector.ROBOT_TUNED_TUBE_RIGHT_OFFSET_CM,
        }
        calibration = {
            "marker_height_cm": 9.0,
            "markers": {
                "4": {
                    "dx": 5.0,
                    "dy": -2.0,
                    "alpha_rad": 0.05,
                }
            },
        }
        estimator = RobotPoseEstimator(marker_detector=FakeRobotMarkerDetector())
        new_pose, new_px, new_observations, new_parallax = estimator.estimate(frame, params, calibration)

        original_extract = detector.extract_robot_marker_observations
        try:
            detector.extract_robot_marker_observations = fake_extract
            legacy_pose, legacy_px, legacy_observations, legacy_parallax = detector.estimate_robot_pose(
                frame,
                params,
                calibration,
                object(),
                object(),
            )
        finally:
            detector.extract_robot_marker_observations = original_extract

        self.assertEqual(set(new_observations), set(legacy_observations))
        np.testing.assert_allclose(new_parallax.camera_center, legacy_parallax.camera_center)
        self.assertEqual(new_px, legacy_px)
        self.assertIsNotNone(new_pose)
        self.assertEqual(new_pose, legacy_pose)

    def test_robot_calibration_collector_matches_spin_center_logic(self) -> None:
        points_4 = [
            (10.0 + math.cos(index * 0.4) * 5.0, 20.0 + math.sin(index * 0.4) * 5.0)
            for index in range(detector.MIN_ROBOT_SPIN_POINTS)
        ]
        points_5 = [
            (30.0 + math.cos(index * 0.4) * 4.0, 40.0 + math.sin(index * 0.4) * 4.0)
            for index in range(detector.MIN_ROBOT_SPIN_POINTS)
        ]
        legacy_runtime = detector.RobotCalibrationRuntime()
        new_runtime = detector.RobotCalibrationRuntime()
        legacy_runtime.collected_points = {4: list(points_4), 5: list(points_5)}
        new_runtime.collected_points = {4: list(points_4), 5: list(points_5)}
        collector = RobotCalibrationCollector()

        self.assertEqual(collector.fit_circle(points_4), detector.fit_circle(points_4))
        self.assertEqual(collector.compute_spin_centers(new_runtime), detector.compute_robot_spin_centers(legacy_runtime))
        self.assertEqual(new_runtime.fitted_centers, legacy_runtime.fitted_centers)
        self.assertEqual(new_runtime.warning, legacy_runtime.warning)

    def test_robot_calibration_scaling_matches_legacy_scaling(self) -> None:
        calibration = {
            "topdown_size": [400, 300],
            "camera_center_x": 200.0,
            "camera_center_y": 150.0,
            "markers": {
                "4": {"dx": 10.0, "dy": 20.0, "origin_x": 30.0, "origin_y": 40.0},
                "5": {"dx": -5.0, "dy": 8.0, "origin_x": 60.0, "origin_y": 70.0},
            },
        }
        collector = RobotCalibrationCollector()
        self.assertEqual(
            collector.scale_robot_calibration_to_topdown(calibration, (800, 600)),
            detector.scale_robot_calibration_to_topdown(calibration, (800, 600)),
        )

    def test_footprint_collision_checker_matches_legacy_checker(self) -> None:
        grid = np.zeros((detector.FIELD_GRID_HEIGHT_CM, detector.FIELD_GRID_WIDTH_CM), dtype=np.uint8)
        grid[60:70, 80:90] = 1
        geometry = detector.robot_geometry_from_params(None)
        pose = detector.HybridPose(40.0, 40.0, 0.2)
        legacy_checker = detector.RobotFootprintCollisionChecker(grid, geometry)
        new_checker = RobotFootprintCollisionChecker(grid, geometry)

        self.assertEqual(new_checker.base_circles, legacy_checker.base_circles)
        self.assertEqual(new_checker.intake_circles, legacy_checker.intake_circles)
        np.testing.assert_allclose(
            np.array(new_checker.oriented_circle_centers(pose, new_checker.base_circles)),
            np.array(legacy_checker.oriented_circle_centers(pose, legacy_checker.base_circles)),
        )
        new_base, new_tube = new_checker.footprint_polygons(pose)
        legacy_base, legacy_tube = legacy_checker.footprint_polygons(pose)
        np.testing.assert_allclose(new_base, legacy_base)
        np.testing.assert_allclose(new_tube, legacy_tube)
        self.assertEqual(new_checker.is_pose_valid(pose), legacy_checker.is_pose_valid(pose))

    def test_path_planners_match_legacy_searches_and_route_tracking(self) -> None:
        grid = np.zeros((detector.FIELD_GRID_HEIGHT_CM, detector.FIELD_GRID_WIDTH_CM), dtype=np.uint8)
        geometry = detector.robot_geometry_from_params(None)
        start_pose = detector.HybridPose(20.0, 20.0, 0.0)
        goal_node = detector.field_metric_cm_to_grid_node((36.0, 20.0))
        config = detector.HybridPlannerConfig(max_expansions=500)
        hybrid = HybridAStarPlanner(config=config)

        self.assertEqual(
            hybrid.search(grid, start_pose, goal_node, geometry, config),
            detector.hybrid_a_star_search(grid, start_pose, goal_node, geometry, config),
        )
        legacy_grid = np.zeros((10, 10), dtype=np.uint8)
        legacy_grid[5, 1:8] = 1
        self.assertEqual(
            LegacyAStarPlanner().search(legacy_grid, (0, 0), (9, 9)),
            detector.a_star_search(legacy_grid, (0, 0), (9, 9)),
        )

        target = detector.PlannedBallTarget(0, "white", 36.0, 20.0, goal_node)
        facade = RoutePlanningFacade(hybrid_config=config)
        self.assertEqual(
            facade.plan_route(grid, [target], start_pose, geometry, config),
            detector.build_greedy_route(grid, [target], start_pose, geometry, config),
        )

        route = [detector.HybridPose(0.0, 0.0, 0.0), detector.HybridPose(10.0, 0.0, 0.0)]
        robot_pose = detector.RobotPose(5.0, 2.0, 0.1, 0.0, 0.0)
        self.assertEqual(
            facade.nearest_route_distance_cm(detector.HybridPose(4.0, 3.0, 0.0), route),
            detector.nearest_route_distance_cm(detector.HybridPose(4.0, 3.0, 0.0), route),
        )
        self.assertEqual(
            facade.compute_route_tracking_error(robot_pose, route),
            detector.compute_route_tracking_error(robot_pose, route),
        )

    def test_wheel_command_controller_matches_legacy_compute_wheel_command(self) -> None:
        error = detector.RouteTrackingError(
            xte_cm=3.0,
            signed_xte_cm=-2.0,
            heading_error_rad=0.35,
            closest_point_cm=(1.0, 2.0),
            segment_heading_rad=0.1,
            segment_index=0,
        )

        self.assertEqual(WheelCommandController().compute(error), detector.compute_wheel_command(error))

    def test_drive_safety_guard_matches_legacy_state_transitions(self) -> None:
        guard = DriveSafetyGuard()
        runtime = detector.DriveRuntime(enabled=True)
        robot_pose = detector.RobotPose(5.0, 30.0, 0.0, 0.0, 0.0)
        route = [detector.HybridPose(0.0, 0.0, 0.0), detector.HybridPose(10.0, 0.0, 0.0)]
        cleared = []

        guard.enforce_xte_guard_before_replan(robot_pose, route, runtime, clear_route_cache=lambda: cleared.append(True))

        self.assertEqual(runtime.state, detector.DriveControlState.REPLANNING)
        self.assertTrue(runtime.suppress_dispatch_this_frame)
        self.assertEqual(cleared, [True])
        self.assertIsNotNone(runtime.last_error)

    def test_udp_wheel_dispatcher_matches_legacy_payload_and_limits(self) -> None:
        class FakeSocket:
            def __init__(self, *_args):
                self.sent = []
                self.blocking = None
                self.closed = False

            def setblocking(self, value):
                self.blocking = value

            def sendto(self, payload, address):
                self.sent.append((payload, address))

            def close(self):
                self.closed = True

        fake_socket = FakeSocket()
        times = iter([10.0, 10.005, 10.05])
        dispatcher = UdpWheelDispatcher(
            "127.0.0.1",
            1234,
            "LR {left:.1f} {right:.1f}",
            socket_factory=lambda *_args: fake_socket,
            time_fn=lambda: next(times),
        )

        self.assertTrue(dispatcher.send_wheel_speeds(100.0, -100.0))
        self.assertTrue(dispatcher.send_wheel_speeds(80.2, -80.2))
        self.assertTrue(dispatcher.send_wheel_speeds(1.0, 2.0, force=True))
        self.assertEqual(fake_socket.sent[0], (b"LR 80.0 -80.0", ("127.0.0.1", 1234)))
        self.assertEqual(fake_socket.sent[1], (b"LR 1.0 2.0", ("127.0.0.1", 1234)))
        self.assertFalse(dispatcher.send_wheel_speeds(float("nan"), 0.0))
        self.assertEqual(dispatcher.last_error, "non-finite wheel command rejected")
        dispatcher.close()
        self.assertTrue(fake_socket.closed)

    def test_debug_renderer_core_outputs_match_legacy_debug_helpers(self) -> None:
        renderer = DebugRenderer()
        frame = np.zeros((80, 120, 3), dtype=np.uint8)
        contour = np.array([[[10, 10]], [[30, 10]], [[30, 30]], [[10, 30]]], dtype=np.int32)
        red_zone = detector.RedZoneDetection(contour, contour.copy(), (10, 10, 20, 20), (20, 20), (20, 20), 400.0)
        white_ball = detector.BallDetection("white", (50, 40), (50, 40), 5, contour, 100.0, 0.9)
        orange_ball = detector.BallDetection("orange", (70, 40), (70, 40), 6, contour, 144.0, 0.8)

        np.testing.assert_array_equal(
            renderer.annotate_camera_frame(frame, [red_zone], [white_ball], [orange_ball], 12.5),
            detector.annotate_camera_frame(frame, [red_zone], [white_ball], [orange_ball], 12.5),
        )

        red_mask = np.zeros((20, 30), dtype=np.uint8)
        white_mask = np.zeros((20, 30), dtype=np.uint8)
        orange_mask = np.zeros((20, 30), dtype=np.uint8)
        red_mask[2:5, 2:5] = 255
        white_mask[6:9, 6:9] = 255
        orange_mask[10:14, 10:14] = 255
        np.testing.assert_array_equal(
            renderer.build_mask_preview(red_mask, white_mask, orange_mask),
            detector.build_mask_preview(red_mask, white_mask, orange_mask),
        )

        left = np.zeros((40, 20, 3), dtype=np.uint8)
        right = np.zeros((20, 10, 3), dtype=np.uint8)
        new_left, new_right = renderer.resize_to_match_height(left, right)
        legacy_left, legacy_right = detector.resize_to_match_height(left, right)
        np.testing.assert_array_equal(new_left, legacy_left)
        np.testing.assert_array_equal(new_right, legacy_right)
        np.testing.assert_array_equal(
            renderer.make_topdown_placeholder("Waiting"),
            detector.make_topdown_placeholder("Waiting"),
        )

    def test_debug_renderer_robot_footprint_matches_legacy_geometry(self) -> None:
        renderer = DebugRenderer()
        pose = detector.HybridPose(40.0, 50.0, 0.4)
        geometry = detector.robot_geometry_from_params(None)
        self.assertEqual(
            renderer.robot_footprint_metric_polygons(pose, geometry),
            detector.robot_footprint_metric_polygons(pose, geometry),
        )

        schematic_new = np.zeros((detector.SCHEMATIC_HEIGHT_PX, detector.SCHEMATIC_WIDTH_PX, 3), dtype=np.uint8)
        schematic_legacy = schematic_new.copy()
        renderer.draw_robot_footprint_snapshot(
            schematic_new,
            pose,
            geometry,
            alpha=0.48,
            base_color=(255, 0, 255),
            intake_color=(0, 255, 255),
            thickness=3,
        )
        detector.draw_robot_footprint_snapshot(
            schematic_legacy,
            pose,
            geometry,
            alpha=0.48,
            base_color=(255, 0, 255),
            intake_color=(0, 255, 255),
            thickness=3,
        )
        np.testing.assert_array_equal(schematic_new, schematic_legacy)


if __name__ == "__main__":
    unittest.main()
