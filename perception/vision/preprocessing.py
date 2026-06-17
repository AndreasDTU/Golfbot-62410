"""Frame preprocessing pipeline for top-down perception.

The ordering here follows the repository vision contract:

1. Undistort
2. Perspective transform
3. Illumination normalization

The class is intentionally independent of OpenCV UI windows.  It returns
structured intermediate frames so the legacy app can keep its current debug
views while we migrate responsibilities one piece at a time.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from perception.vision.calibration import HomographyCalibrationResult, HomographyCalibrator, UndistortionProvider
from config import CameraConfig
from perception.vision.models import CalibrationState, CameraGroundProjection


@dataclass(frozen=True)
class PreprocessedFrame:
    """Intermediate outputs from the preprocessing stage."""

    undistorted: np.ndarray
    topdown: np.ndarray | None
    normalized: np.ndarray | None
    calibration_state: CalibrationState
    transform_matrix: np.ndarray | None
    camera_ground_projection: CameraGroundProjection | None
    homography_result: HomographyCalibrationResult | None = None


class FramePreprocessor:
    """Run raw frames through undistortion, top-down warp, and normalization."""

    def __init__(
        self,
        undistortion_provider: UndistortionProvider,
        homography_calibrator: HomographyCalibrator,
        camera_config: CameraConfig | None = None,
        normalize_illumination: bool = False,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid_size: tuple[int, int] = (8, 8),
    ) -> None:
        self.undistortion_provider = undistortion_provider
        self.homography_calibrator = homography_calibrator
        self.camera = camera_config or homography_calibrator.camera
        self.normalize_illumination_enabled = normalize_illumination
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid_size = clahe_tile_grid_size

    def undistort(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Run the undistortion stage."""
        return self.undistortion_provider.undistort(frame_bgr)

    def update_homography(
        self,
        undistorted_frame: np.ndarray,
        use_aruco: bool = True,
    ) -> HomographyCalibrationResult | None:
        """Refresh automatic homography calibration when requested."""
        if not use_aruco:
            return None
        if self.homography_calibrator.calibration_state in (
            CalibrationState.CALIBRATING_MANUAL,
            CalibrationState.CALIBRATED_MANUAL,
        ):
            return None
        return self.homography_calibrator.calibrate_from_aruco_frame(undistorted_frame)

    def warp_topdown(self, undistorted_frame: np.ndarray) -> np.ndarray | None:
        """Apply the active perspective transform, if available."""
        return self.homography_calibrator.warp(undistorted_frame)

    def normalize_illumination(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Normalize illumination with CLAHE on the Lab L channel.

        This implements the contract-friendly normalization stage while keeping
        color channels in BGR for downstream code that still expects BGR frames.
        """
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=self.clahe_tile_grid_size,
        )
        normalized_l = clahe.apply(l_channel)
        normalized_lab = cv2.merge((normalized_l, a_channel, b_channel))
        return cv2.cvtColor(normalized_lab, cv2.COLOR_LAB2BGR)

    def process(
        self,
        frame_bgr: np.ndarray,
        use_aruco: bool = True,
        normalize_illumination: bool | None = None,
    ) -> PreprocessedFrame:
        """Run the preprocessing pipeline in contract order."""
        undistorted = self.undistort(frame_bgr)
        live_gray = self.homography_calibrator.update_frame_context(undistorted)
        if self.homography_calibrator.calibration_state in (
            CalibrationState.CALIBRATING_MANUAL,
            CalibrationState.CALIBRATED_MANUAL,
        ):
            self.homography_calibrator.update_manual_anchor_tracking(live_gray)
        homography_result = self.update_homography(undistorted, use_aruco=use_aruco)
        topdown = self.warp_topdown(undistorted)
        should_normalize = self.normalize_illumination_enabled if normalize_illumination is None else normalize_illumination
        normalized = self.normalize_illumination(topdown) if should_normalize and topdown is not None else topdown
        camera_ground_projection = self.homography_calibrator.project_principal_point_to_topdown(
            self.undistortion_provider.maps.undistorted_camera_matrix,
        )
        return PreprocessedFrame(
            undistorted=undistorted,
            topdown=topdown,
            normalized=normalized,
            calibration_state=self.homography_calibrator.calibration_state,
            transform_matrix=self.homography_calibrator.transform_matrix,
            camera_ground_projection=camera_ground_projection,
            homography_result=homography_result,
        )
