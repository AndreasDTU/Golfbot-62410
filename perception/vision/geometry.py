"""Geometry utilities for deterministic top-down vision.

The classes in this module contain coordinate and parallax math only.  They do
not own OpenCV windows, mutable app state, camera capture, or robot control.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from config import CameraConfig, FieldConfig, WindowConfig
from perception.vision.models import CameraGroundProjection, ParallaxConfig


class CoordinateMapper:
    """Convert between top-down pixels, field centimeters, grid nodes, and UI pixels."""

    def __init__(
        self,
        field_config: FieldConfig | None = None,
        camera_config: CameraConfig | None = None,
        window_config: WindowConfig | None = None,
    ) -> None:
        self.field = field_config or FieldConfig()
        self.camera = camera_config or CameraConfig()
        self.window = window_config or WindowConfig()

    def order_points(self, points: Any) -> np.ndarray:
        """Order 4 selected corners as top-left, top-right, bottom-right, bottom-left."""
        corners = np.array(points, dtype=np.float32).reshape(4, 2)
        sums = corners.sum(axis=1)
        diffs = np.diff(corners, axis=1).reshape(4)

        top_left = corners[np.argmin(sums)]
        bottom_right = corners[np.argmax(sums)]
        top_right = corners[np.argmin(diffs)]
        bottom_left = corners[np.argmax(diffs)]
        return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)

    @property
    def _topdown_world_span_cm(self) -> tuple[float, float]:
        """World extent (cm) covered by the top-down warp, including the margin border."""
        margin = self.camera.topdown_margin_cm
        return (self.field.width_cm + 2.0 * margin, self.field.height_cm + 2.0 * margin)

    def destination_corners(self, size: tuple[int, int] | None = None) -> np.ndarray:
        """Build the playable-field rectangle (inset by the margin) for a perspective warp."""
        width, height = size or self.camera.topdown_warp_size
        span_w, span_h = self._topdown_world_span_cm
        margin = self.camera.topdown_margin_cm
        x0 = margin * (width - 1) / span_w
        x1 = (margin + self.field.width_cm) * (width - 1) / span_w
        y0 = margin * (height - 1) / span_h
        y1 = (margin + self.field.height_cm) * (height - 1) / span_h
        return np.array(
            [
                [x0, y0],
                [x1, y0],
                [x1, y1],
                [x0, y1],
            ],
            dtype=np.float32,
        )

    def build_manual_topdown_transform(
        self,
        points: Any,
        size: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """Compute a manual perspective transform from 4 selected corners."""
        return cv2.getPerspectiveTransform(
            self.order_points(points),
            self.destination_corners(size),
        )

    def topdown_px_to_field_cm(self, point_px: np.ndarray | tuple[float, float]) -> tuple[float, float]:
        """Convert fixed top-down pixels to bottom-left-origin field centimeters."""
        width, height = self.camera.topdown_warp_size
        span_w, span_h = self._topdown_world_span_cm
        margin = self.camera.topdown_margin_cm
        x_cm = float(point_px[0]) * span_w / max(1, width - 1) - margin
        y_cm = self.field.height_cm - (float(point_px[1]) * span_h / max(1, height - 1) - margin)
        return float(x_cm), float(y_cm)

    def field_cm_to_topdown_pixel(self, point_cm: tuple[float, float]) -> tuple[float, float]:
        """Convert bottom-left-origin field centimeters to fixed top-down pixels."""
        width, height = self.camera.topdown_warp_size
        span_w, span_h = self._topdown_world_span_cm
        margin = self.camera.topdown_margin_cm
        x_px = (float(point_cm[0]) + margin) * (width - 1) / span_w
        y_px = (self.field.height_cm - float(point_cm[1]) + margin) * (height - 1) / span_h
        return float(x_px), float(y_px)

    def field_cm_to_topdown_px_unflipped(self, x_cm: float, y_cm: float) -> tuple[float, float]:
        """Map field centimeters to top-down pixels without flipping the Y axis.

        This matches the legacy ArUco destination-point helper where marker
        world coordinates are expressed in the image-oriented top-down plane.
        """
        width, height = self.camera.topdown_warp_size
        span_w, span_h = self._topdown_world_span_cm
        margin = self.camera.topdown_margin_cm
        scale_x = (width - 1) / span_w
        scale_y = (height - 1) / span_h
        return (x_cm + margin) * scale_x, (y_cm + margin) * scale_y

    def map_point_between_frames(
        self,
        point: tuple[int, int],
        source_size: tuple[int, int],
        destination_size: tuple[int, int],
    ) -> tuple[int, int]:
        """Map a point between image coordinate systems by simple linear scaling."""
        src_width, src_height = source_size
        dst_width, dst_height = destination_size
        x = int(point[0] * dst_width / max(1, src_width))
        y = int(point[1] * dst_height / max(1, src_height))
        return x, y

    def pixel_to_field_cm(
        self,
        point: tuple[int, int],
        source_size: tuple[int, int],
    ) -> tuple[float, float]:
        """Convert top-down source pixels to bottom-left-origin field centimeters."""
        src_width, src_height = source_size
        span_w, span_h = self._topdown_world_span_cm
        margin = self.camera.topdown_margin_cm
        x_cm = float(point[0]) * span_w / max(1, src_width - 1) - margin
        y_cm = self.field.height_cm - (float(point[1]) * span_h / max(1, src_height - 1) - margin)
        return (
            float(np.clip(x_cm, 0.0, self.field.width_cm)),
            float(np.clip(y_cm, 0.0, self.field.height_cm)),
        )

    def pixel_float_to_field_cm(
        self,
        point: np.ndarray,
        source_size: tuple[int, int],
    ) -> tuple[float, float]:
        """Convert floating-point top-down pixels to bottom-left-origin field centimeters."""
        return self.pixel_to_field_cm(
            (int(round(float(point[0]))), int(round(float(point[1])))),
            source_size,
        )

    def field_metric_cm_to_grid_node(self, point_cm: tuple[float, float]) -> tuple[int, int]:
        """Convert bottom-left metric coordinates to top-left grid nodes."""
        x_cm, y_cm = point_cm
        return (
            int(np.clip(round(x_cm), 0, self.field.grid_width_cm - 1)),
            int(np.clip(round(self.field.height_cm - y_cm), 0, self.field.grid_height_cm - 1)),
        )

    def field_metric_cm_to_schematic(self, point_cm: tuple[float, float]) -> tuple[int, int]:
        """Convert bottom-left metric coordinates to schematic pixels."""
        x_cm, y_cm = point_cm
        x_px = int(round(x_cm * (self.window.schematic_width_px - 1) / max(1.0, self.field.width_cm)))
        y_px = int(
            round((self.field.height_cm - y_cm) * (self.window.schematic_height_px - 1) / max(1.0, self.field.height_cm))
        )
        return (
            int(np.clip(x_px, 0, self.window.schematic_width_px - 1)),
            int(np.clip(y_px, 0, self.window.schematic_height_px - 1)),
        )

    def schematic_to_field_metric_cm(self, point_px: tuple[int, int]) -> tuple[float, float]:
        """Convert schematic pixels back to bottom-left metric coordinates."""
        x_px, y_px = point_px
        x_cm = float(x_px) * self.field.width_cm / max(1, self.window.schematic_width_px - 1)
        y_cm = self.field.height_cm - (
            float(y_px) * self.field.height_cm / max(1, self.window.schematic_height_px - 1)
        )
        return (
            float(np.clip(x_cm, 0.0, self.field.width_cm)),
            float(np.clip(y_cm, 0.0, self.field.height_cm)),
        )

    def source_point_to_field_cm(
        self,
        point: tuple[int, int],
        source_size: tuple[int, int],
    ) -> tuple[int, int]:
        """Map a source-frame pixel to the 1 cm occupancy-grid coordinate space."""
        src_width, src_height = source_size
        x = int(round(point[0] * (self.field.grid_width_cm - 1) / max(1, src_width - 1)))
        y = int(round(point[1] * (self.field.grid_height_cm - 1) / max(1, src_height - 1)))
        return (
            int(np.clip(x, 0, self.field.grid_width_cm - 1)),
            int(np.clip(y, 0, self.field.grid_height_cm - 1)),
        )

    def field_cm_to_schematic(self, point_cm: tuple[int, int]) -> tuple[int, int]:
        """Map a 1 cm grid coordinate to the schematic window."""
        return self.map_point_between_frames(
            point_cm,
            (self.field.grid_width_cm, self.field.grid_height_cm),
            self.window.schematic_size_px,
        )

    def contour_to_field_grid(self, contour: np.ndarray, source_size: tuple[int, int]) -> np.ndarray:
        """Convert a contour from source pixels to the 1 cm occupancy grid."""
        mapped_points = [
            self.source_point_to_field_cm((int(point[0][0]), int(point[0][1])), source_size)
            for point in contour
        ]
        return np.array(mapped_points, dtype=np.int32).reshape((-1, 1, 2))

    def project_principal_point_to_topdown(
        self,
        camera_matrix: np.ndarray,
        homography: np.ndarray | None,
    ) -> CameraGroundProjection | None:
        """Map the undistorted camera principal point through the active homography."""
        if camera_matrix.shape != (3, 3):
            return None
        if homography is None or homography.shape != (3, 3):
            return None

        principal_point = np.array(
            [[[float(camera_matrix[0, 2]), float(camera_matrix[1, 2])]]],
            dtype=np.float32,
        )
        projected = cv2.perspectiveTransform(principal_point, homography.astype(np.float64)).reshape(2)
        if not np.all(np.isfinite(projected)):
            return None

        return CameraGroundProjection(
            principal_point_px=principal_point.reshape(2).astype(np.float32),
            camera_center_px=projected.astype(np.float32),
        )


class ParallaxCorrector:
    """Apply deterministic radial parallax correction around a camera center."""

    def correct_point(
        self,
        pixel_coord: tuple[int, int],
        z_object_cm: float,
        h_cam_cm: float,
        z_calib_cm: float,
        camera_center_pixels: tuple[float, float],
    ) -> tuple[int, int]:
        """Correct radial displacement caused by object height relative to the warp plane."""
        camera_center_x, camera_center_y = camera_center_pixels
        denominator = h_cam_cm - z_calib_cm
        if abs(denominator) < 1e-6:
            return int(round(pixel_coord[0])), int(round(pixel_coord[1]))
        scale = (h_cam_cm - z_object_cm) / denominator

        x_real = camera_center_x + (pixel_coord[0] - camera_center_x) * scale
        y_real = camera_center_y + (pixel_coord[1] - camera_center_y) * scale
        return int(round(x_real)), int(round(y_real))

    def correct_contour(
        self,
        contour: np.ndarray,
        z_object_cm: float,
        h_cam_cm: float,
        z_calib_cm: float,
        camera_center_pixels: tuple[float, float],
    ) -> np.ndarray:
        """Apply parallax correction point-by-point for contour-based overlays."""
        corrected_points = [
            self.correct_point(
                pixel_coord=(int(point[0][0]), int(point[0][1])),
                z_object_cm=z_object_cm,
                h_cam_cm=h_cam_cm,
                z_calib_cm=z_calib_cm,
                camera_center_pixels=camera_center_pixels,
            )
            for point in contour
        ]
        return np.array(corrected_points, dtype=np.int32).reshape((-1, 1, 2))

    def correct_point_float(self, point: np.ndarray, config: ParallaxConfig) -> np.ndarray:
        """Project an elevated floating-point image point down to the calibration plane."""
        denominator = config.camera_height_cm - config.calibration_plane_height_cm
        if abs(denominator) < 1e-6:
            return point.astype(np.float32)

        scale = (config.camera_height_cm - config.marker_height_cm) / denominator
        return (config.camera_center + (point - config.camera_center) * scale).astype(np.float32)
