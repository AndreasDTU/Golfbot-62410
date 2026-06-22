"""Robot marker localization and calibration collection.

This module extracts robot-specific ArUco detection, parallax projection, pose
estimation, and spin-calibration state management from the legacy detector.
The classes are deterministic and side-effect free except where explicitly
saving calibration JSON.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import cv2
import numpy as np

from localization.models import RobotCalibrationRuntime, RobotGeometry, RobotMarkerObservation, RobotPose
from perception.vision.calibration import HomographyCalibrator
from config import CameraConfig, FieldConfig, PlannerConfig, PoseSmoothingConfig, RobotCalibrationConfig, RobotGeometryConfig
from perception.vision.geometry import CoordinateMapper, ParallaxCorrector
from perception.vision.models import ParallaxConfig


def normalize_angle(angle: float) -> float:
    """Normalize an angle to [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def image_yaw_rotation_matrix(angle: float) -> np.ndarray:
    """Rotate top-down image vectors for yaw measured from +Y with image Y down."""
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([[c, s], [-s, c]], dtype=np.float32)


def image_yaw_to_field_heading(angle: float) -> float:
    """Convert image yaw to field heading, where +X is 0 and +Y is pi/2."""
    return normalize_angle(math.atan2(-math.cos(angle), math.sin(angle)))


def marker_yaw_from_ground_corners(ground_corners: np.ndarray) -> float:
    """Return marker yaw in top-down image coordinates; 0 points toward image +Y."""
    pts = ground_corners.reshape(4, 2).astype(np.float32)
    top_mid = (pts[0] + pts[1]) * 0.5
    bottom_mid = (pts[2] + pts[3]) * 0.5
    marker_forward = bottom_mid - top_mid
    return float(math.atan2(marker_forward[0], marker_forward[1]))


class RobotMarkerDetector:
    """Detect robot ArUco markers and project their corners to the ground plane."""

    def __init__(
        self,
        marker_ids: tuple[int, ...] | None = None,
        dictionary: object | None = None,
        detector_or_parameters: object | None = None,
        parallax_corrector: ParallaxCorrector | None = None,
        camera_config: CameraConfig | None = None,
    ) -> None:
        self.camera_config = camera_config or CameraConfig()
        self.marker_ids = marker_ids or RobotGeometryConfig().marker_ids
        if dictionary is None or detector_or_parameters is None:
            dictionary, detector_or_parameters = HomographyCalibrator.build_aruco_detector()
        self.dictionary = dictionary
        self.detector_or_parameters = detector_or_parameters
        self.parallax_corrector = parallax_corrector or ParallaxCorrector()

    def detect_markers(self, frame: np.ndarray) -> tuple[list[np.ndarray], np.ndarray | None]:
        """Detect ArUco markers in the already warped top-down frame."""
        if self.dictionary is None or self.detector_or_parameters is None:
            return [], None
        return HomographyCalibrator.detect_aruco_markers(frame, self.dictionary, self.detector_or_parameters)

    def extract_observations(
        self,
        frame: np.ndarray,
        parallax_config: ParallaxConfig,
    ) -> dict[int, RobotMarkerObservation]:
        """Detect robot markers and immediately project their geometry to ground."""
        observations: dict[int, RobotMarkerObservation] = {}
        corners, ids = self.detect_markers(frame)
        if ids is None:
            return observations

        wanted = set(self.marker_ids)
        for marker_corners, raw_id in zip(corners, ids.flatten().tolist()):
            marker_id = int(raw_id)
            if marker_id not in wanted:
                continue

            pts = marker_corners.reshape(4, 2).astype(np.float32)
            ground_pts = np.array(
                [self.parallax_corrector.correct_point_float(point, parallax_config) for point in pts],
                dtype=np.float32,
            )
            observations[marker_id] = RobotMarkerObservation(
                marker_id=marker_id,
                center=pts.mean(axis=0).astype(np.float32),
                ground_center=ground_pts.mean(axis=0).astype(np.float32),
                corners=pts,
                ground_corners=ground_pts,
                yaw_rad=marker_yaw_from_ground_corners(ground_pts),
            )
        return observations


class RobotPoseEstimator:
    """Estimate robot origin, heading, and pickup tube pose from marker observations."""

    def __init__(
        self,
        field_config: FieldConfig | None = None,
        robot_config: RobotGeometryConfig | None = None,
        mapper: CoordinateMapper | None = None,
        marker_detector: RobotMarkerDetector | None = None,
        smoothing_config: PoseSmoothingConfig | None = None,
        planner_config: PlannerConfig | None = None,
    ) -> None:
        self.field = field_config or FieldConfig()
        self.robot_config = robot_config or RobotGeometryConfig()
        self.planner_config = planner_config or PlannerConfig()
        self.mapper = mapper or CoordinateMapper(self.field)
        self.marker_detector = marker_detector or RobotMarkerDetector(marker_ids=self.robot_config.marker_ids)
        self.smoothing = smoothing_config or PoseSmoothingConfig()
        self._reset_smoothing()

    def _reset_smoothing(self) -> None:
        """Drop the EMA filter state so the next pose re-seeds without lag.

        Called on (re)acquisition gaps: if the robot vanishes and reappears
        (or teleports while occluded), we want to snap to the fresh measurement
        rather than crawl toward it across several frames.
        """
        self._smoothed_x_cm: float | None = None
        self._smoothed_y_cm: float | None = None
        self._smoothed_cos: float | None = None
        self._smoothed_sin: float | None = None

    def _smooth_pose(
        self,
        x_cm: float,
        y_cm: float,
        heading_rad: float,
    ) -> tuple[float, float, float]:
        """Blend a raw pose with the running estimate via EMA.

        Position is a straight linear EMA.  Heading is filtered as a (cos, sin)
        unit vector so the average wraps correctly across the +/-pi seam instead
        of jumping the long way around.
        """
        if not self.smoothing.enabled:
            return x_cm, y_cm, heading_rad

        cos_h = math.cos(heading_rad)
        sin_h = math.sin(heading_rad)

        if self._smoothed_x_cm is None:
            # First sample after a gap: seed directly so there is no startup lag.
            self._smoothed_x_cm = x_cm
            self._smoothed_y_cm = y_cm
            self._smoothed_cos = cos_h
            self._smoothed_sin = sin_h
            return x_cm, y_cm, heading_rad

        pos_alpha = self.smoothing.position_alpha
        head_alpha = self.smoothing.heading_alpha
        self._smoothed_x_cm += pos_alpha * (x_cm - self._smoothed_x_cm)
        self._smoothed_y_cm += pos_alpha * (y_cm - self._smoothed_y_cm)
        self._smoothed_cos += head_alpha * (cos_h - self._smoothed_cos)
        self._smoothed_sin += head_alpha * (sin_h - self._smoothed_sin)
        smoothed_heading = math.atan2(self._smoothed_sin, self._smoothed_cos)
        return self._smoothed_x_cm, self._smoothed_y_cm, normalize_angle(smoothed_heading)

    def robot_geometry_from_params(self, params: dict[str, object] | None) -> RobotGeometry:
        """Read live robot geometry, falling back to measured defaults."""
        params = params or {}
        return RobotGeometry(
            width_cm=float(params.get("robot_width_cm", self.robot_config.tuned_footprint_width_cm)),
            front_cm=float(params.get("robot_front_cm", self.robot_config.tuned_footprint_front_from_origin_cm)),
            rear_cm=float(params.get("robot_rear_cm", self.robot_config.tuned_footprint_rear_from_origin_cm)),
            tube_forward_cm=float(params.get("tube_forward_cm", self.robot_config.tuned_tube_offset_cm)),
            tube_right_cm=float(params.get("tube_right_cm", self.robot_config.tuned_tube_right_offset_cm)),
            tube_width_cm=float(params.get("tube_width_cm", self.planner_config.tube_width_cm)),
            mouth_radius_cm=float(params.get("mouth_radius_cm", self.planner_config.mouth_radius_cm)),
            unload_extension_cm=float(params.get("unload_extension_cm", self.planner_config.unload_extension_cm)),
            pipe_diameter_cm=float(params.get("pipe_diameter_cm", self.planner_config.pipe_diameter_cm)),
        )

    def parallax_config_from_live_params(
        self,
        params: dict[str, object],
        calibration: dict[str, Any] | None,
    ) -> ParallaxConfig:
        """Build marker parallax config from live UI/calibration values."""
        marker_height_cm = self.robot_config.marker_height_cm
        if calibration is not None and isinstance(calibration.get("marker_height_cm"), (int, float)):
            marker_height_cm = float(calibration["marker_height_cm"])

        return ParallaxConfig(
            marker_height_cm=marker_height_cm,
            camera_height_cm=float(params["h_cam_cm"]),
            calibration_plane_height_cm=float(params["z_calib_cm"]),
            camera_center=np.array(
                [float(params["camera_center_x"]), float(params["camera_center_y"])],
                dtype=np.float32,
            ),
        )

    def robot_origin_from_observation(
        self,
        observation: RobotMarkerObservation,
        marker_calibration: dict[str, float],
    ) -> tuple[np.ndarray, float]:
        """Apply saved offset after parallax correction to get robot origin."""
        offset = np.array([marker_calibration["dx"], marker_calibration["dy"]], dtype=np.float32)
        true_image_heading = normalize_angle(observation.yaw_rad - float(marker_calibration["alpha_rad"]))
        rotated_offset = image_yaw_rotation_matrix(true_image_heading) @ offset
        return observation.ground_center - rotated_offset, true_image_heading

    def estimate(
        self,
        frame_bgr: np.ndarray,
        params: dict[str, object],
        calibration: dict[str, Any] | None,
    ) -> tuple[
        RobotPose | None,
        tuple[float, float] | None,
        dict[int, RobotMarkerObservation],
        ParallaxConfig,
    ]:
        """Estimate the robot pose from the current warped frame and live state."""
        parallax_config = self.parallax_config_from_live_params(params, calibration)
        observations = self.marker_detector.extract_observations(frame_bgr, parallax_config)

        if calibration is None:
            self._reset_smoothing()
            return None, None, observations, parallax_config

        origins: list[np.ndarray] = []
        field_headings: list[float] = []
        for marker_id, observation in observations.items():
            marker_config = calibration.get("markers", {}).get(str(marker_id))
            if marker_config is None:
                continue
            origin_px, true_image_heading = self.robot_origin_from_observation(observation, marker_config)
            origins.append(origin_px)
            field_headings.append(image_yaw_to_field_heading(true_image_heading))

        if not origins:
            self._reset_smoothing()
            return None, None, observations, parallax_config

        origin = np.mean(np.array(origins, dtype=np.float32), axis=0)
        source_height, source_width = frame_bgr.shape[:2]
        x_cm, y_cm = self.mapper.pixel_float_to_field_cm(origin, (source_width, source_height))
        heading_rad = math.atan2(
            sum(math.sin(angle) for angle in field_headings),
            sum(math.cos(angle) for angle in field_headings),
        )
        heading_rad = normalize_angle(
            heading_rad
            + self.robot_config.forward_heading_offset_rad
            + float(params.get("heading_tuning_rad", 0.0))
        )
        # Low-pass the primitive DOF before deriving the tube pose, so the tube
        # stays geometrically consistent with the smoothed origin/heading.
        x_cm, y_cm, heading_rad = self._smooth_pose(x_cm, y_cm, heading_rad)
        geometry = self.robot_geometry_from_params(params)
        forward = (math.cos(heading_rad), math.sin(heading_rad))
        right = (math.sin(heading_rad), -math.cos(heading_rad))
        tube_x_cm = x_cm + forward[0] * geometry.tube_forward_cm + right[0] * geometry.tube_right_cm
        tube_y_cm = y_cm + forward[1] * geometry.tube_forward_cm + right[1] * geometry.tube_right_cm
        return (
            RobotPose(
                x_cm=x_cm,
                y_cm=y_cm,
                heading_rad=heading_rad,
                tube_x_cm=tube_x_cm,
                tube_y_cm=tube_y_cm,
            ),
            (float(origin[0]), float(origin[1])),
            observations,
            parallax_config,
        )


class RobotCalibrationCollector:
    """Manage robot spin calibration point collection and JSON persistence."""

    def __init__(
        self,
        robot_config: RobotGeometryConfig | None = None,
        calibration_config: RobotCalibrationConfig | None = None,
    ) -> None:
        self.robot_config = robot_config or RobotGeometryConfig()
        self.calibration_config = calibration_config or RobotCalibrationConfig()

    def scale_robot_calibration_to_topdown(
        self,
        calibration: dict[str, Any],
        topdown_size: tuple[int, int],
    ) -> dict[str, Any]:
        """Scale saved top-down pixel calibration values if warp size differs."""
        stored_size = calibration.get("topdown_size")
        if not isinstance(stored_size, list) or len(stored_size) != 2:
            return calibration

        stored_width = float(stored_size[0])
        stored_height = float(stored_size[1])
        target_width = float(topdown_size[0])
        target_height = float(topdown_size[1])
        if stored_width <= 0.0 or stored_height <= 0.0:
            return calibration
        if int(stored_width) == topdown_size[0] and int(stored_height) == topdown_size[1]:
            return calibration

        scale_x = target_width / stored_width
        scale_y = target_height / stored_height
        scaled = json.loads(json.dumps(calibration))
        for key, scale in (("camera_center_x", scale_x), ("camera_center_y", scale_y)):
            if isinstance(scaled.get(key), (int, float)):
                scaled[key] = float(scaled[key]) * scale

        for marker_config in scaled.get("markers", {}).values():
            for key, scale in (("dx", scale_x), ("origin_x", scale_x), ("dy", scale_y), ("origin_y", scale_y)):
                if isinstance(marker_config.get(key), (int, float)):
                    marker_config[key] = float(marker_config[key]) * scale
        scaled["topdown_size"] = [topdown_size[0], topdown_size[1]]
        print(
            f"Scaled robot calibration from {(int(stored_width), int(stored_height))} to {topdown_size}. "
            "Recalibrate in this detector for best accuracy.",
            file=sys.stderr,
        )
        return scaled

    def load_robot_calibration(
        self,
        path: Path,
        topdown_size: tuple[int, int],
    ) -> dict[str, Any] | None:
        """Load marker offsets from JSON if all requested marker IDs are present."""
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            calibration = json.load(handle)

        markers = calibration.get("markers", {})
        if not all(str(marker_id) in markers for marker_id in self.robot_config.marker_ids):
            print(f"Robot calibration missing marker IDs {self.robot_config.marker_ids}: {path}", file=sys.stderr)
            return None
        return self.scale_robot_calibration_to_topdown(calibration, topdown_size)

    @staticmethod
    def fit_circle(points: list[tuple[float, float]]) -> tuple[float, float, float]:
        """Fit a deterministic enclosing circle to collected spin points."""
        pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        (x, y), radius = cv2.minEnclosingCircle(pts)
        return float(x), float(y), float(radius)

    @staticmethod
    def ellipse_ratio(points: list[tuple[float, float]]) -> float | None:
        """Estimate how circular the spin path is; >1 means ellipse-like."""
        if len(points) < 5:
            return None
        pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        _center, axes, _angle = cv2.fitEllipse(pts)
        minor = max(1e-6, float(min(axes)))
        major = float(max(axes))
        return major / minor

    def compute_spin_centers(self, runtime: RobotCalibrationRuntime) -> bool:
        """Finalize spin collection by fitting one circle center per marker.

        Tolerant of a missing marker: any marker with enough collected points is
        fitted, and the calibration only fails when *no* marker qualifies.  This
        lets the robot self-calibrate (or refresh) from a single visible marker
        when the other is occluded during the spin.
        """
        runtime.fitted_centers.clear()
        runtime.ellipse_ratios.clear()
        ellipse_warnings: list[str] = []
        skipped: list[int] = []

        for marker_id in self.robot_config.marker_ids:
            points = runtime.collected_points.get(marker_id, [])
            if len(points) < self.calibration_config.min_robot_spin_points:
                skipped.append(marker_id)
                continue
            xc, yc, _radius = self.fit_circle(points)
            runtime.fitted_centers[marker_id] = (xc, yc)
            ratio = self.ellipse_ratio(points)
            if ratio is not None:
                runtime.ellipse_ratios[marker_id] = ratio
                if ratio > self.calibration_config.ellipse_warning_ratio:
                    ellipse_warnings.append(f"ID {marker_id} ellipse ratio {ratio:.2f}")

        if not runtime.fitted_centers:
            runtime.warning = (
                f"Need {self.calibration_config.min_robot_spin_points} spin points "
                f"for at least one marker {self.robot_config.marker_ids}."
            )
            return False

        messages: list[str] = []
        if skipped:
            messages.append(f"Skipped marker(s) {skipped}: too few spin points")
        if ellipse_warnings:
            messages.append(
                "WARNING: Elliptical spin path (" + ", ".join(ellipse_warnings) + "). "
                "Check floor slip or homography."
            )
        runtime.warning = " | ".join(messages)
        return True

    @staticmethod
    def _read_existing_calibration(path: Path) -> dict[str, Any]:
        """Return the parsed calibration JSON, or an empty dict if absent/corrupt."""
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def build_robot_calibration(
        self,
        runtime: RobotCalibrationRuntime,
        observations: dict[int, RobotMarkerObservation],
        parallax_config: ParallaxConfig,
        topdown_size: tuple[int, int],
        existing: dict[str, Any] | None = None,
        geometry: RobotGeometry | None = None,
        heading_tuning_rad: float = 0.0,
    ) -> dict[str, Any]:
        """Build a calibration dict in memory (no disk write).

        Merges with ``existing``: markers that were not re-fit this run (e.g. one
        marker was occluded) keep their previously saved offsets, and the
        geometry tuning block is preserved unless ``geometry`` is given.  Only
        markers that were both fitted *and* currently observed are updated, since
        computing the offset needs a live observation at the aligned pose.
        """
        existing = existing or {}
        markers: dict[str, dict[str, float]] = dict(existing.get("markers", {}))
        for marker_id, origin in runtime.fitted_centers.items():
            observation = observations.get(marker_id)
            if observation is None:
                continue
            dx = float(observation.ground_center[0] - origin[0])
            dy = float(observation.ground_center[1] - origin[1])
            alpha_rad = float(observation.yaw_rad)
            markers[str(marker_id)] = {
                "dx": dx,
                "dy": dy,
                "alpha_rad": alpha_rad,
                "alpha_deg": math.degrees(alpha_rad),
                "origin_x": float(origin[0]),
                "origin_y": float(origin[1]),
                "ellipse_ratio": float(runtime.ellipse_ratios.get(marker_id, 0.0)),
            }

        calibration = {
            "version": 1,
            "created_unix": time.time(),
            "marker_ids": list(self.robot_config.marker_ids),
            "marker_height_cm": float(parallax_config.marker_height_cm),
            "camera_height_cm": float(parallax_config.camera_height_cm),
            "calibration_plane_height_cm": float(parallax_config.calibration_plane_height_cm),
            "camera_center_x": float(parallax_config.camera_center[0]),
            "camera_center_y": float(parallax_config.camera_center[1]),
            "topdown_size": [int(topdown_size[0]), int(topdown_size[1])],
            "markers": markers,
        }

        geometry_block = existing.get("geometry")
        if geometry is not None:
            geometry_block = self.geometry_to_dict(geometry, heading_tuning_rad)
        if isinstance(geometry_block, dict) and geometry_block:
            calibration["geometry"] = geometry_block

        return calibration

    def save_robot_calibration(
        self,
        path: Path,
        runtime: RobotCalibrationRuntime,
        observations: dict[int, RobotMarkerObservation],
        parallax_config: ParallaxConfig,
        topdown_size: tuple[int, int],
        geometry: RobotGeometry | None = None,
        heading_tuning_rad: float = 0.0,
    ) -> dict[str, Any]:
        """Build (merging with any on-disk calibration) and persist to ``path``."""
        existing = self._read_existing_calibration(path)
        calibration = self.build_robot_calibration(
            runtime,
            observations,
            parallax_config,
            topdown_size,
            existing=existing,
            geometry=geometry,
            heading_tuning_rad=heading_tuning_rad,
        )
        path.write_text(json.dumps(calibration, indent=2, sort_keys=True), encoding="utf-8")
        return calibration

    @staticmethod
    def geometry_to_dict(geometry: RobotGeometry, heading_tuning_rad: float = 0.0) -> dict[str, float]:
        """Serialize live robot body geometry (cm) plus heading tuning for JSON."""
        return {
            "width_cm": float(geometry.width_cm),
            "front_cm": float(geometry.front_cm),
            "rear_cm": float(geometry.rear_cm),
            "tube_forward_cm": float(geometry.tube_forward_cm),
            "tube_right_cm": float(geometry.tube_right_cm),
            "tube_width_cm": float(geometry.tube_width_cm),
            "mouth_radius_cm": float(geometry.mouth_radius_cm),
            "unload_extension_cm": float(geometry.unload_extension_cm),
            "pipe_diameter_cm": float(geometry.pipe_diameter_cm),
            "heading_tuning_rad": float(heading_tuning_rad),
            "heading_tuning_deg": math.degrees(float(heading_tuning_rad)),
        }

    def save_geometry_tuning(
        self,
        path: Path,
        geometry: RobotGeometry,
        heading_tuning_rad: float = 0.0,
    ) -> dict[str, Any]:
        """Persist only the geometry/heading tuning, preserving spin offsets.

        Lets the operator nudge the virtual body to match the physical robot and
        save without re-running a spin.
        """
        calibration = self._read_existing_calibration(path)
        calibration["geometry"] = self.geometry_to_dict(geometry, heading_tuning_rad)
        path.write_text(json.dumps(calibration, indent=2, sort_keys=True), encoding="utf-8")
        return calibration

    @staticmethod
    def apply_geometry_to_params(
        params: dict[str, object],
        geometry_block: dict[str, Any] | None,
    ) -> dict[str, object]:
        """Seed live detector params from a saved geometry block, in place."""
        if not isinstance(geometry_block, dict):
            return params
        mapping = {
            "width_cm": "robot_width_cm",
            "front_cm": "robot_front_cm",
            "rear_cm": "robot_rear_cm",
            "tube_forward_cm": "tube_forward_cm",
            "tube_right_cm": "tube_right_cm",
            "tube_width_cm": "tube_width_cm",
            "mouth_radius_cm": "mouth_radius_cm",
            "unload_extension_cm": "unload_extension_cm",
            "pipe_diameter_cm": "pipe_diameter_cm",
            "heading_tuning_rad": "heading_tuning_rad",
        }
        for source_key, param_key in mapping.items():
            value = geometry_block.get(source_key)
            if isinstance(value, (int, float)):
                params[param_key] = float(value)
        return params
