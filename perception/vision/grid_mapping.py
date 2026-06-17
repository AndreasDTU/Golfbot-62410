"""Occupancy-grid construction for path planning."""

from __future__ import annotations

import cv2
import numpy as np

from config import FieldConfig, RobotGeometryConfig
from perception.vision.geometry import CoordinateMapper
from perception.vision.models import RedZoneDetection


class OccupancyGridBuilder:
    """Build a 1 cm binary occupancy grid from detected red zones."""

    def __init__(
        self,
        field_config: FieldConfig | None = None,
        robot_config: RobotGeometryConfig | None = None,
        mapper: CoordinateMapper | None = None,
    ) -> None:
        self.field = field_config or FieldConfig()
        self.robot = robot_config or RobotGeometryConfig()
        self.mapper = mapper or CoordinateMapper(self.field)

    def contour_to_field_grid(self, contour: np.ndarray, source_size: tuple[int, int]) -> np.ndarray:
        """Convert a contour from source pixels to the 1 cm occupancy grid."""
        return self.mapper.contour_to_field_grid(contour, source_size)

    def build(
        self,
        frame_shape: tuple[int, int, int],
        red_zones: list[RedZoneDetection],
        dilate_for_legacy: bool = True,
    ) -> np.ndarray:
        """Build the binary occupancy grid used by legacy A* and Hybrid A*.

        Legacy 2D A* callers receive the historical circular robot-radius
        dilation by default.  Hybrid A* callers can pass
        ``dilate_for_legacy=False`` to keep raw red-zone occupancy and delegate
        clearance checks to an oriented robot footprint.
        """
        source_height, source_width = frame_shape[:2]
        grid = np.zeros((self.field.grid_height_cm, self.field.grid_width_cm), dtype=np.uint8)

        for zone in red_zones:
            grid_contour = self.contour_to_field_grid(zone.corrected_contour, (source_width, source_height))
            cv2.fillPoly(grid, [grid_contour], 1)

        if not dilate_for_legacy:
            return (grid > 0).astype(np.uint8)

        kernel_size = max(1, int(2 * self.robot.radius_cm + 1))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        dilated = cv2.dilate(grid, kernel, iterations=1)
        return (dilated > 0).astype(np.uint8)
