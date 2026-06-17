"""Detection and segmentation components for top-down perception.

YOLO is intentionally quarantined in :class:`YoloBallDetector`.  Importing this
module does not import ``ultralytics``; the dependency is loaded only when that
adapter is constructed.  This keeps the rest of the vision package classical-CV
friendly and easy to test.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from config import FieldConfig
from perception.vision.geometry import ParallaxCorrector
from perception.vision.models import BallDetection, HSVRange, RedZoneDetection


class BallDetector(Protocol):
    """Interface for ball detectors used by the perception pipeline."""

    def detect(
        self,
        frame_bgr: np.ndarray,
        params: dict[str, object],
        camera_center_pixels: tuple[float, float],
    ) -> tuple[list[BallDetection], list[BallDetection], dict[str, np.ndarray]]:
        """Return white detections, orange detections, and per-label masks."""


class MaskCleaner:
    """Deterministic morphology cleanup for binary masks."""

    def cleanup(self, mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """Apply open/close morphology to reduce single-pixel noise."""
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        return cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)


class RedZoneDetector:
    """Detect red avoidance zones with HSV thresholding and contour extraction."""

    def __init__(
        self,
        field_config: FieldConfig | None = None,
        parallax_corrector: ParallaxCorrector | None = None,
        mask_cleaner: MaskCleaner | None = None,
    ) -> None:
        self.field = field_config or FieldConfig()
        self.parallax_corrector = parallax_corrector or ParallaxCorrector()
        self.mask_cleaner = mask_cleaner or MaskCleaner()

    @staticmethod
    def contour_center(contour: np.ndarray) -> tuple[int, int]:
        """Compute a contour centroid with a bounding-box fallback."""
        moments = cv2.moments(contour)
        if moments["m00"] <= 0.0:
            x, y, width, height = cv2.boundingRect(contour)
            return (x + width // 2, y + height // 2)
        return (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]))

    def detect(
        self,
        frame_bgr: np.ndarray,
        params: dict[str, object],
        camera_center_pixels: tuple[float, float],
    ) -> tuple[list[RedZoneDetection], np.ndarray]:
        """Detect red avoidance zones from a top-down image."""
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        red_1 = params["red_1"]
        red_2 = params["red_2"]
        if not isinstance(red_1, HSVRange) or not isinstance(red_2, HSVRange):
            raise TypeError("params['red_1'] and params['red_2'] must be HSVRange instances.")

        mask_1 = cv2.inRange(hsv, red_1.lower, red_1.upper)
        mask_2 = cv2.inRange(hsv, red_2.lower, red_2.upper)
        combined = cv2.bitwise_or(mask_1, mask_2)
        combined = self.mask_cleaner.cleanup(combined, kernel_size=5)

        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: list[RedZoneDetection] = []

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < float(params["red_min_area"]):
                continue

            x, y, width, height = cv2.boundingRect(contour)
            center = self.contour_center(contour)
            detections.append(
                RedZoneDetection(
                    contour=contour,
                    corrected_contour=self.parallax_corrector.correct_contour(
                        contour=contour,
                        z_object_cm=self.field.floor_height_cm,
                        h_cam_cm=float(params["h_cam_cm"]),
                        z_calib_cm=float(params["z_calib_cm"]),
                        camera_center_pixels=camera_center_pixels,
                    ),
                    bounding_box=(x, y, width, height),
                    center=center,
                    corrected_center=self.parallax_corrector.correct_point(
                        pixel_coord=center,
                        z_object_cm=self.field.floor_height_cm,
                        h_cam_cm=float(params["h_cam_cm"]),
                        z_calib_cm=float(params["z_calib_cm"]),
                        camera_center_pixels=camera_center_pixels,
                    ),
                    area=area,
                )
            )

        return detections, combined


class YoloBallDetector:
    """YOLO ball detector adapter preserving the legacy detector output shape."""

    def __init__(
        self,
        model_path: Path,
        field_config: FieldConfig | None = None,
        parallax_corrector: ParallaxCorrector | None = None,
        model: object | None = None,
    ) -> None:
        self.field = field_config or FieldConfig()
        self.parallax_corrector = parallax_corrector or ParallaxCorrector()
        self.model_path = model_path
        if model is None:
            from ultralytics import YOLO

            self.model = YOLO(str(model_path))
        else:
            self.model = model

    def detect(
        self,
        frame_bgr: np.ndarray,
        params: dict[str, object],
        camera_center_pixels: tuple[float, float],
    ) -> tuple[list[BallDetection], list[BallDetection], dict[str, np.ndarray]]:
        """Detect white and orange balls while preserving legacy masks and sorting."""
        white_detections: list[BallDetection] = []
        orange_detections: list[BallDetection] = []
        white_mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
        orange_mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
        label_by_class_name = {
            "branca": "white",
            "laranja": "orange",
        }
        confidence_threshold = float(params["yolo_confidence"])
        min_area = float(params["yolo_min_area"])
        max_area = max(min_area, float(params["yolo_max_area"]))

        results = self.model(frame_bgr, verbose=False)[0]
        if results.boxes is not None:
            for box in results.boxes:
                confidence = float(box.conf[0].cpu())
                if confidence < confidence_threshold:
                    continue

                class_index = int(box.cls[0].cpu())
                class_name = str(results.names[class_index]).strip().lower()
                label = label_by_class_name.get(class_name)
                if label is None:
                    continue

                x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].cpu().tolist())
                x1_i = int(round(np.clip(x1, 0, frame_bgr.shape[1] - 1)))
                y1_i = int(round(np.clip(y1, 0, frame_bgr.shape[0] - 1)))
                x2_i = int(round(np.clip(x2, 0, frame_bgr.shape[1] - 1)))
                y2_i = int(round(np.clip(y2, 0, frame_bgr.shape[0] - 1)))
                if x2_i <= x1_i or y2_i <= y1_i:
                    continue

                area = float((x2_i - x1_i) * (y2_i - y1_i))
                if area < min_area or area > max_area:
                    continue

                center = ((x1_i + x2_i) // 2, (y1_i + y2_i) // 2)
                radius_px = max(2, int(round(max(x2_i - x1_i, y2_i - y1_i) * 0.5)))
                contour = np.array(
                    [
                        [[x1_i, y1_i]],
                        [[x2_i, y1_i]],
                        [[x2_i, y2_i]],
                        [[x1_i, y2_i]],
                    ],
                    dtype=np.int32,
                )
                detection = BallDetection(
                    label=label,
                    center=center,
                    corrected_center=self.parallax_corrector.correct_point(
                        pixel_coord=center,
                        z_object_cm=self.field.ball_height_cm,
                        h_cam_cm=float(params["h_cam_cm"]),
                        z_calib_cm=float(params["z_calib_cm"]),
                        camera_center_pixels=camera_center_pixels,
                    ),
                    radius_px=radius_px,
                    contour=contour,
                    area=area,
                    circularity=confidence,
                )

                if label == "white":
                    white_detections.append(detection)
                    cv2.circle(white_mask, center, radius_px, 255, -1, cv2.LINE_AA)
                else:
                    orange_detections.append(detection)
                    cv2.circle(orange_mask, center, radius_px, 255, -1, cv2.LINE_AA)

        white_detections.sort(key=lambda ball: (ball.center[1], ball.center[0]))
        orange_detections.sort(key=lambda ball: (ball.center[1], ball.center[0]))
        masks = {"white": white_mask, "orange": orange_mask}
        return white_detections, orange_detections, masks
