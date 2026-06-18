"""Facade orchestration for the top-down vision pipeline.

``VisionPipeline`` composes the already-extracted preprocessing, detection,
tracking, and grid-mapping components behind one frame-level API.  The class is
dependency-injection friendly so tests, legacy adapters, and future classical
ball detectors can be swapped in without changing the orchestration contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from perception.vision.calibration import HomographyCalibrator, UndistortionProvider
from config import AppConfig, FieldConfig
from perception.vision.detection import BallDetector, RedZoneDetector, YoloBallDetector
from perception.vision.geometry import CoordinateMapper
from perception.vision.grid_mapping import OccupancyGridBuilder
from perception.vision.models import BallDetection, HSVRange, RedZoneDetection, SmoothedBallCoordinate
from perception.vision.preprocessing import FramePreprocessor, PreprocessedFrame
from perception.vision.tracking import BallCoordinateSmoother


@dataclass(frozen=True)
class VisionFrameResult:
    """Structured outputs from processing one frame."""

    raw_frame: np.ndarray
    preprocessed: PreprocessedFrame
    frame_for_detection: np.ndarray | None
    red_zones: list[RedZoneDetection]
    red_mask: np.ndarray
    white_balls: list[BallDetection]
    orange_balls: list[BallDetection]
    ball_masks: dict[str, np.ndarray]
    smoothed_ball_coordinates: list[SmoothedBallCoordinate]
    occupancy_grid: np.ndarray | None
    params: dict[str, object] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def all_balls(self) -> list[BallDetection]:
        """Return white and orange detections in the same order used for smoothing."""
        return self.white_balls + self.orange_balls


class VisionPipeline:
    """Frame-level facade for preprocessing, detection, tracking, and grid mapping."""

    def __init__(
        self,
        app_config: AppConfig | None = None,
        preprocessor: FramePreprocessor | None = None,
        ball_detector: BallDetector | None = None,
        red_zone_detector: RedZoneDetector | None = None,
        ball_smoother: BallCoordinateSmoother | None = None,
        occupancy_grid_builder: OccupancyGridBuilder | None = None,
        mapper: CoordinateMapper | None = None,
        build_legacy_dilated_grid: bool = False,
        normalize_illumination: bool | None = None,
    ) -> None:
        if app_config is None:
            app_config = AppConfig.from_repo_root(Path(__file__).resolve().parents[1])
        self.config = app_config
        self.mapper = mapper or CoordinateMapper(app_config.field, app_config.camera, app_config.windows)
        self.preprocessor = preprocessor or self._build_default_preprocessor(normalize_illumination)
        self.ball_detector = ball_detector or self._build_default_ball_detector()
        self.red_zone_detector = red_zone_detector or RedZoneDetector(app_config.field)
        self.ball_smoother = ball_smoother or BallCoordinateSmoother(self.mapper)
        self.occupancy_grid_builder = occupancy_grid_builder or OccupancyGridBuilder(
            app_config.field,
            app_config.robot,
            self.mapper,
        )
        self.build_legacy_dilated_grid = build_legacy_dilated_grid

    def _build_default_preprocessor(self, normalize_illumination: bool | None) -> FramePreprocessor:
        provider = UndistortionProvider(self.config.paths.calibration_file)
        calibrator = HomographyCalibrator(self.config.field, self.config.camera, self.mapper)
        return FramePreprocessor(
            provider,
            calibrator,
            self.config.camera,
            normalize_illumination=bool(normalize_illumination),
        )

    def _build_default_ball_detector(self) -> BallDetector:
        model_path = self.config.paths.yolo_model_path
        if model_path is None:
            raise ValueError("A YOLO model path or injected BallDetector is required.")
        if not model_path.exists():
            local_model = Path(__file__).resolve().parents[1] / "tools" / model_path.name
            if local_model.exists():
                model_path = local_model
        return YoloBallDetector(model_path, self.config.field)

    def default_params(self) -> dict[str, object]:
        """Build detector params from typed config defaults.

        The shape mirrors the legacy ``read_hsv_ranges`` dictionary so extracted
        components can be wired into the old app incrementally.
        """
        defaults = self.config.detection.trackbar_defaults(self.config.field, self.config.robot)
        camera_center_px = self.mapper.field_cm_to_topdown_pixel(
            (
                float(defaults["cam_center_x"]),
                float(defaults["cam_center_y"]),
            )
        )
        return {
            "red_1": HSVRange(
                lower=np.array(
                    [defaults["red1_h_min"], defaults["red_s_min"], defaults["red_v_min"]],
                    dtype=np.uint8,
                ),
                upper=np.array(
                    [defaults["red1_h_max"], defaults["red_s_max"], defaults["red_v_max"]],
                    dtype=np.uint8,
                ),
            ),
            "red_2": HSVRange(
                lower=np.array(
                    [defaults["red2_h_min"], defaults["red_s_min"], defaults["red_v_min"]],
                    dtype=np.uint8,
                ),
                upper=np.array(
                    [defaults["red2_h_max"], defaults["red_s_max"], defaults["red_v_max"]],
                    dtype=np.uint8,
                ),
            ),
            "red_min_area": float(defaults["red_min_area"]),
            "yolo_confidence": float(defaults["yolo_conf_pct"]) / 100.0,
            "yolo_min_area": float(defaults["yolo_min_area"]),
            "yolo_max_area": float(defaults["yolo_max_area"]),
            "h_cam_cm": float(defaults["cam_height_cm"]),
            "z_calib_cm": float(defaults["calib_z_cm"]),
            "camera_center_x": float(camera_center_px[0]),
            "camera_center_y": float(camera_center_px[1]),
            "camera_center_x_cm": float(defaults["cam_center_x"]),
            "camera_center_y_cm": float(defaults["cam_center_y"]),
        }

    def process(
        self,
        frame: np.ndarray,
        params: dict[str, object] | None = None,
        use_aruco: bool = True,
        normalize_illumination: bool | None = None,
        dilate_for_legacy: bool | None = None,
        detect_red_zones: bool = True,
        extra_red_zones: list[RedZoneDetection] | None = None,
    ) -> VisionFrameResult:
        """Process one frame through the extracted vision components.

        ``detect_red_zones=False`` skips HSV red-zone segmentation (e.g. when the
        central cross is specified manually instead).  ``extra_red_zones`` are
        appended to the detected ones and flow into both the occupancy grid and
        the overlays, so a manually placed obstacle becomes a real avoidance zone.
        """
        effective_params = self.default_params() if params is None else dict(params)
        preprocessed = self.preprocessor.process(
            frame,
            use_aruco=use_aruco,
            normalize_illumination=normalize_illumination,
        )
        frame_for_detection = preprocessed.normalized
        if frame_for_detection is None:
            red_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            ball_masks = {
                "white": np.zeros(frame.shape[:2], dtype=np.uint8),
                "orange": np.zeros(frame.shape[:2], dtype=np.uint8),
            }
            return VisionFrameResult(
                raw_frame=frame,
                preprocessed=preprocessed,
                frame_for_detection=None,
                red_zones=[],
                red_mask=red_mask,
                white_balls=[],
                orange_balls=[],
                ball_masks=ball_masks,
                smoothed_ball_coordinates=[],
                occupancy_grid=None,
                params=effective_params,
                metadata={"status": "missing_topdown_transform"},
            )

        camera_center_pixels = (
            float(effective_params["camera_center_x"]),
            float(effective_params["camera_center_y"]),
        )
        if detect_red_zones:
            red_zones, red_mask = self.red_zone_detector.detect(
                frame_for_detection,
                effective_params,
                camera_center_pixels,
            )
        else:
            red_zones = []
            red_mask = np.zeros(frame_for_detection.shape[:2], dtype=np.uint8)
        if extra_red_zones:
            red_zones = list(red_zones) + list(extra_red_zones)
        white_balls, orange_balls, ball_masks = self.ball_detector.detect(
            frame_for_detection,
            effective_params,
            camera_center_pixels,
        )
        all_balls = white_balls + orange_balls
        smoothed = self.ball_smoother.update(all_balls, frame_for_detection.shape)
        occupancy_grid = self.occupancy_grid_builder.build(
            frame_for_detection.shape,
            red_zones,
            dilate_for_legacy=self.build_legacy_dilated_grid if dilate_for_legacy is None else dilate_for_legacy,
        )
        return VisionFrameResult(
            raw_frame=frame,
            preprocessed=preprocessed,
            frame_for_detection=frame_for_detection,
            red_zones=red_zones,
            red_mask=red_mask,
            white_balls=white_balls,
            orange_balls=orange_balls,
            ball_masks=ball_masks,
            smoothed_ball_coordinates=smoothed,
            occupancy_grid=occupancy_grid,
            params=effective_params,
            metadata={"status": "ok"},
        )
