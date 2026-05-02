"""Camera undistortion and top-down homography calibration.

This module extracts calibration math from the legacy detector while keeping
OpenCV window/UI drawing out of the calibration classes.  The legacy app can
continue to own keyboard and mouse events, then pass selected points or frames
into these deterministic helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from vision.config import CameraConfig, FieldConfig
from vision.geometry import CoordinateMapper
from vision.models import CalibrationState, CameraGroundProjection


@dataclass(frozen=True)
class UndistortionMaps:
    """Precomputed fisheye undistortion maps for one calibrated image size."""

    undistorted_camera_matrix: np.ndarray
    image_size: tuple[int, int]
    map1: np.ndarray
    map2: np.ndarray


class UndistortionProvider:
    """Load and apply fisheye undistortion state from a calibration ``.npz`` file."""

    def __init__(self, calibration_file: Path, balance: float = 0.0) -> None:
        self.calibration_file = calibration_file
        self.balance = balance
        self._maps: UndistortionMaps | None = None

    @property
    def maps(self) -> UndistortionMaps:
        if self._maps is None:
            self._maps = self.load_maps(self.calibration_file, self.balance)
        return self._maps

    @staticmethod
    def load_calibration_image_size(calibration_file: Path) -> tuple[int, int]:
        """Read the calibration image size so live capture matches the saved model."""
        data = np.load(str(calibration_file))
        return tuple(int(value) for value in data["image_size"])

    @staticmethod
    def load_maps(calibration_file: Path, balance: float) -> UndistortionMaps:
        """Precompute fisheye undistortion state once for a stream."""
        data = np.load(str(calibration_file))
        camera_matrix = data["K"].astype(np.float64)
        dist_coeffs = data["D"].astype(np.float64)
        image_size = tuple(int(value) for value in data["image_size"])
        undistorted_camera_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            camera_matrix,
            dist_coeffs,
            image_size,
            np.eye(3, dtype=np.float64),
            balance=balance,
        )
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            camera_matrix,
            dist_coeffs,
            np.eye(3, dtype=np.float64),
            undistorted_camera_matrix,
            image_size,
            cv2.CV_32FC1,
        )
        return UndistortionMaps(
            undistorted_camera_matrix=undistorted_camera_matrix.astype(np.float64),
            image_size=image_size,
            map1=map1,
            map2=map2,
        )

    def validate_frame_size(self, frame_bgr: np.ndarray) -> None:
        """Raise if the frame size differs from the saved calibration size."""
        frame_size = (int(frame_bgr.shape[1]), int(frame_bgr.shape[0]))
        if frame_size != self.maps.image_size:
            raise ValueError(
                f"Billedstoerrelse {frame_size} matcher ikke kalibreringsstoerrelsen "
                f"{self.maps.image_size}. Kalibrering og drift skal bruge samme oploesning."
            )

    def undistort(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Apply the precomputed fisheye remap to one frame."""
        self.validate_frame_size(frame_bgr)
        undistorted = cv2.remap(
            frame_bgr,
            self.maps.map1,
            self.maps.map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        if not np.any(undistorted):
            raise ValueError(
                "Undistort gav et helt sort billede. Kalibreringsparametrene er sandsynligvis ustabile "
                "eller passer ikke til dette billede."
            )
        return undistorted


@dataclass
class HomographyCalibrationResult:
    """Result of a homography update attempt."""

    transform_matrix: np.ndarray | None
    calibration_state: CalibrationState
    marker_centers: dict[int, np.ndarray] = field(default_factory=dict)
    corners: list[np.ndarray] = field(default_factory=list)
    ids: np.ndarray | None = None


class HomographyCalibrator:
    """Manage manual and ArUco-based top-down homography calibration."""

    def __init__(
        self,
        field_config: FieldConfig | None = None,
        camera_config: CameraConfig | None = None,
        mapper: CoordinateMapper | None = None,
    ) -> None:
        self.field = field_config or FieldConfig()
        self.camera = camera_config or CameraConfig()
        self.mapper = mapper or CoordinateMapper(self.field, self.camera)
        self.transform_matrix: np.ndarray | None = None
        self.calibration_state = CalibrationState.NEEDS_CALIBRATION
        self.manual_points: list[tuple[int, int]] = []
        self.cursor: tuple[int, int] = (0, 0)
        self.frame_size: tuple[int, int] = (0, 0)
        self.latest_gray_frame: np.ndarray | None = None
        self.anchor_gray_frame: np.ndarray | None = None
        self.anchor_points: np.ndarray | None = None
        self.current_tracked_points: np.ndarray | None = None
        self.tracked_point_valid: np.ndarray = np.zeros(4, dtype=bool)
        self.tracked_point_errors: np.ndarray = np.full(4, np.inf, dtype=np.float32)
        self.lk_fb_max_error_px = 2.0
        self.lk_ema_alpha = 0.2
        self.lk_params = {
            "winSize": self.camera.lk_win_size,
            "maxLevel": self.camera.lk_max_level,
            "criteria": (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                self.camera.lk_criteria_count,
                self.camera.lk_criteria_epsilon,
            ),
        }
        self.latest_aruco_centers: dict[int, np.ndarray] = {}
        self.aruco_dictionary, self.aruco_detector = self.build_aruco_detector()

    @property
    def aruco_available(self) -> bool:
        return self.aruco_dictionary is not None and self.aruco_detector is not None

    @staticmethod
    def build_aruco_detector() -> tuple[object | None, object | None]:
        """Create an ArUco detector that works across OpenCV versions."""
        if not hasattr(cv2, "aruco"):
            return None, None

        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        if hasattr(cv2.aruco, "DetectorParameters"):
            parameters = cv2.aruco.DetectorParameters()
        else:
            parameters = cv2.aruco.DetectorParameters_create()

        if hasattr(cv2.aruco, "ArucoDetector"):
            return dictionary, cv2.aruco.ArucoDetector(dictionary, parameters)
        return dictionary, parameters

    @staticmethod
    def detect_aruco_markers(
        frame: np.ndarray,
        dictionary: object,
        detector_or_parameters: object,
    ) -> tuple[list[np.ndarray], np.ndarray | None]:
        """Detect ArUco markers in an already-undistorted frame."""
        if hasattr(detector_or_parameters, "detectMarkers"):
            corners, ids, _rejected = detector_or_parameters.detectMarkers(frame)
        else:
            corners, ids, _rejected = cv2.aruco.detectMarkers(frame, dictionary, parameters=detector_or_parameters)
        return corners, ids

    @staticmethod
    def marker_center(corners: np.ndarray) -> np.ndarray:
        """Compute the center of one detected marker from its four corners."""
        return corners.reshape(4, 2).mean(axis=0).astype(np.float32)

    def extract_required_marker_centers(
        self,
        corners: list[np.ndarray],
        ids: np.ndarray | None,
    ) -> dict[int, np.ndarray]:
        """Keep only the marker centers needed for automatic homography calibration."""
        centers: dict[int, np.ndarray] = {}
        if ids is None:
            return centers

        required = set(self.camera.required_aruco_ids)
        for marker_id, marker_corners in zip(ids.flatten().tolist(), corners):
            if marker_id in required:
                centers[int(marker_id)] = self.marker_center(marker_corners)
        return centers

    def aruco_destination_points(self) -> np.ndarray:
        """Build target marker-center positions in the cropped top-down pixel space."""
        total_offset_cm = self.camera.wall_thickness_cm + self.camera.marker_outer_offset_cm
        world_points_cm = (
            (-total_offset_cm, -total_offset_cm),
            (self.field.width_cm + total_offset_cm, -total_offset_cm),
            (self.field.width_cm + total_offset_cm, self.field.height_cm + total_offset_cm),
            (-total_offset_cm, self.field.height_cm + total_offset_cm),
        )
        return np.array(
            [self.mapper.field_cm_to_topdown_px_unflipped(x_cm, y_cm) for x_cm, y_cm in world_points_cm],
            dtype=np.float32,
        )

    def topdown_field_corners(self) -> np.ndarray:
        """Return the playable field rectangle in the warped top-down pixel plane."""
        return self.mapper.destination_corners(self.camera.topdown_warp_size)

    def build_auto_topdown_transform(self, marker_centers: dict[int, np.ndarray]) -> np.ndarray | None:
        """Compute the perspective transform when all required markers are visible."""
        required = self.camera.required_aruco_ids
        if not all(marker_id in marker_centers for marker_id in required):
            return None

        src_points = np.array(
            [
                marker_centers[0],
                marker_centers[1],
                marker_centers[3],
                marker_centers[2],
            ],
            dtype=np.float32,
        )
        return cv2.getPerspectiveTransform(src_points, self.aruco_destination_points())

    def build_manual_topdown_transform(self, points: object) -> np.ndarray:
        """Compute the manual perspective transform from four selected corners."""
        return self.mapper.build_manual_topdown_transform(points, self.camera.topdown_warp_size)

    def start_manual_calibration(self) -> None:
        """Switch to manual calibration and clear pending points."""
        self.manual_points.clear()
        self.transform_matrix = None
        self.latest_aruco_centers.clear()
        self.reset_manual_tracking()
        self.calibration_state = CalibrationState.CALIBRATING_MANUAL

    def start_auto_calibration(self) -> None:
        """Switch to ArUco calibration and clear manual points."""
        self.manual_points.clear()
        self.transform_matrix = None
        self.latest_aruco_centers.clear()
        self.reset_manual_tracking()
        self.calibration_state = CalibrationState.NEEDS_CALIBRATION

    def clear_manual_points(self) -> None:
        """Clear selected points without forcing auto mode."""
        self.manual_points.clear()
        self.transform_matrix = None
        self.reset_manual_tracking()
        if self.calibration_state in (CalibrationState.CALIBRATING_MANUAL, CalibrationState.CALIBRATED_MANUAL):
            self.calibration_state = CalibrationState.CALIBRATING_MANUAL
        else:
            self.calibration_state = CalibrationState.NEEDS_CALIBRATION

    def reset_manual_tracking(self) -> None:
        """Reset Lucas-Kanade anchor/tracked point state."""
        self.anchor_gray_frame = None
        self.anchor_points = None
        self.current_tracked_points = None
        self.tracked_point_valid = np.zeros(4, dtype=bool)
        self.tracked_point_errors = np.full(4, np.inf, dtype=np.float32)

    def update_frame_context(self, undistorted_frame: np.ndarray) -> np.ndarray:
        """Refresh frame size and latest grayscale frame used by manual tracking."""
        live_gray = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2GRAY)
        self.latest_gray_frame = live_gray
        self.frame_size = (int(undistorted_frame.shape[1]), int(undistorted_frame.shape[0]))
        if self.cursor == (0, 0):
            self.cursor = (self.frame_size[0] // 2, self.frame_size[1] // 2)
        return live_gray

    def initialize_manual_anchor_tracking(self) -> bool:
        """Anchor the manual corner tracker to the current undistorted grayscale frame."""
        if len(self.manual_points) != 4 or self.latest_gray_frame is None:
            return False

        anchor_points = np.array(self.manual_points, dtype=np.float32).reshape(4, 2)
        self.anchor_gray_frame = self.latest_gray_frame.copy()
        self.anchor_points = anchor_points
        self.current_tracked_points = anchor_points.copy()
        self.tracked_point_valid = np.ones(4, dtype=bool)
        self.tracked_point_errors = np.zeros(4, dtype=np.float32)
        self.transform_matrix = self.build_manual_topdown_transform(self.current_tracked_points)
        self.calibration_state = CalibrationState.CALIBRATED_MANUAL
        return True

    def update_manual_anchor_tracking(self, live_gray: np.ndarray) -> None:
        """Track manual corners from the immutable anchor frame with FB validation."""
        if (
            self.calibration_state != CalibrationState.CALIBRATED_MANUAL
            or self.anchor_gray_frame is None
            or self.anchor_points is None
            or self.current_tracked_points is None
        ):
            return

        forward_pts, forward_status, _forward_err = cv2.calcOpticalFlowPyrLK(
            self.anchor_gray_frame,
            live_gray,
            self.anchor_points.reshape(-1, 1, 2),
            None,
            **self.lk_params,
        )
        if forward_pts is None or forward_status is None:
            self.tracked_point_valid = np.zeros(4, dtype=bool)
            self.tracked_point_errors = np.full(4, np.inf, dtype=np.float32)
            return

        recovered_pts, backward_status, _backward_err = cv2.calcOpticalFlowPyrLK(
            live_gray,
            self.anchor_gray_frame,
            forward_pts,
            None,
            **self.lk_params,
        )
        if recovered_pts is None or backward_status is None:
            self.tracked_point_valid = np.zeros(4, dtype=bool)
            self.tracked_point_errors = np.full(4, np.inf, dtype=np.float32)
            return

        recovered_flat = recovered_pts.reshape(4, 2)
        forward_flat = forward_pts.reshape(4, 2)
        anchor_flat = self.anchor_points.reshape(4, 2)
        fb_errors = np.linalg.norm(anchor_flat - recovered_flat, axis=1).astype(np.float32)
        valid = (
            (forward_status.reshape(4) == 1)
            & (backward_status.reshape(4) == 1)
            & (fb_errors < self.lk_fb_max_error_px)
            & np.isfinite(fb_errors)
            & np.all(np.isfinite(forward_flat), axis=1)
        )

        if np.any(valid):
            current = self.current_tracked_points.reshape(4, 2)
            current[valid] = (1.0 - self.lk_ema_alpha) * current[valid] + self.lk_ema_alpha * forward_flat[valid]
            self.current_tracked_points = current.astype(np.float32)
            self.transform_matrix = self.build_manual_topdown_transform(self.current_tracked_points)

        self.tracked_point_valid = valid
        self.tracked_point_errors = fb_errors

    def add_manual_point(self, point: tuple[int, int]) -> np.ndarray | None:
        """Add one manual point and return a transform once four points are available."""
        if len(self.manual_points) >= 4:
            return self.transform_matrix

        self.manual_points.append((int(point[0]), int(point[1])))
        if len(self.manual_points) == 4:
            if not self.initialize_manual_anchor_tracking():
                self.transform_matrix = self.build_manual_topdown_transform(self.manual_points)
                self.calibration_state = CalibrationState.CALIBRATED_MANUAL
        else:
            self.calibration_state = CalibrationState.CALIBRATING_MANUAL
        return self.transform_matrix

    def set_manual_points(self, points: list[tuple[int, int]]) -> np.ndarray:
        """Replace all manual points and compute the manual transform."""
        if len(points) != 4:
            raise ValueError("Manual top-down calibration requires exactly 4 points.")
        self.manual_points = [(int(x), int(y)) for x, y in points]
        self.transform_matrix = self.build_manual_topdown_transform(self.manual_points)
        self.calibration_state = CalibrationState.CALIBRATED_MANUAL
        return self.transform_matrix

    def calibrate_from_aruco_frame(self, undistorted_frame: np.ndarray) -> HomographyCalibrationResult:
        """Attempt automatic homography calibration from visible ArUco markers."""
        if not self.aruco_available:
            return HomographyCalibrationResult(
                transform_matrix=self.transform_matrix,
                calibration_state=self.calibration_state,
            )

        corners, ids = self.detect_aruco_markers(
            undistorted_frame,
            self.aruco_dictionary,
            self.aruco_detector,
        )
        self.latest_aruco_centers = self.extract_required_marker_centers(corners, ids)
        auto_transform = self.build_auto_topdown_transform(self.latest_aruco_centers)
        if auto_transform is not None:
            self.transform_matrix = auto_transform
            self.calibration_state = CalibrationState.CALIBRATED_AUTO
            self.manual_points.clear()

        return HomographyCalibrationResult(
            transform_matrix=self.transform_matrix,
            calibration_state=self.calibration_state,
            marker_centers=dict(self.latest_aruco_centers),
            corners=corners,
            ids=ids,
        )

    def project_principal_point_to_topdown(
        self,
        camera_matrix: np.ndarray,
    ) -> CameraGroundProjection | None:
        """Map the undistorted camera principal point through the active homography."""
        return self.mapper.project_principal_point_to_topdown(camera_matrix, self.transform_matrix)

    def warp(self, undistorted_frame: np.ndarray) -> np.ndarray | None:
        """Warp an undistorted frame into the configured fixed top-down rectangle."""
        if self.transform_matrix is None:
            return None
        return cv2.warpPerspective(undistorted_frame, self.transform_matrix, self.camera.topdown_warp_size)
