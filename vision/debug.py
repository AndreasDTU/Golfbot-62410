"""Debug rendering helpers for top-down vision and route planning."""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from pathfinding.models import HybridPose
from pathfinding.planner import HybridAStarPlanner
from robot.control import WheelCommandController, route_checkpoint_indices
from robot.localization import RobotPoseEstimator
from robot.models import (
    DriveRuntime,
    RobotCalibrationPhase,
    RobotCalibrationRuntime,
    RobotGeometry,
    RobotMarkerObservation,
    RobotPose,
)
from vision.config import DriveConfig, FieldConfig, RobotGeometryConfig, WindowConfig
from vision.geometry import CoordinateMapper
from vision.models import BallDetection, RedZoneDetection, SmoothedBallCoordinate


class DebugRenderer:
    """Draw camera overlays, masks, schematic panels, and route debug geometry."""

    def __init__(
        self,
        field_config: FieldConfig | None = None,
        window_config: WindowConfig | None = None,
        robot_config: RobotGeometryConfig | None = None,
        drive_config: DriveConfig | None = None,
        mapper: CoordinateMapper | None = None,
    ) -> None:
        self.field = field_config or FieldConfig()
        self.window = window_config or WindowConfig()
        self.robot_config = robot_config or RobotGeometryConfig()
        self.drive_config = drive_config or DriveConfig()
        self.mapper = mapper or CoordinateMapper(self.field, window_config=self.window)

    def robot_geometry_from_params(self, params: dict[str, object] | None) -> RobotGeometry:
        """Read live robot geometry, falling back to tuned defaults."""
        return RobotPoseEstimator(self.field, self.robot_config, self.mapper).robot_geometry_from_params(params)

    def route_velocity_color(self, distance_to_goal_cm: float) -> tuple[int, int, int]:
        """Map the profiled route speed to a BGR heatmap color."""
        target = WheelCommandController.target_speed_for_distance(distance_to_goal_cm, self.drive_config)
        creep = max(0.0, float(self.drive_config.creep_speed_pct))
        cruise = max(creep + 1e-6, float(self.drive_config.base_speed_pct))
        ratio = float(np.clip((target - creep) / (cruise - creep), 0.0, 1.0))
        if ratio < 0.5:
            local = ratio / 0.5
            red = int(round(255 * local))
            green = 255
        else:
            local = (ratio - 0.5) / 0.5
            red = 255
            green = int(round(255 * (1.0 - local)))
        return (0, green, red)

    def draw_velocity_profile_route(
        self,
        schematic: np.ndarray,
        route_points_cm: list[HybridPose],
        route_pickup_poses_cm: list[HybridPose] | None = None,
    ) -> None:
        """Draw each route segment with the same distance-based speed profile used by control."""
        if len(route_points_cm) < 2:
            return
        checkpoints = route_checkpoint_indices(route_points_cm, route_pickup_poses_cm, include_final=True)
        if not checkpoints:
            checkpoints = [len(route_points_cm) - 1]
        for segment_index, (start, end) in enumerate(zip(route_points_cm[:-1], route_points_cm[1:])):
            start_px = self.mapper.field_metric_cm_to_schematic((start.x_cm, start.y_cm))
            end_px = self.mapper.field_metric_cm_to_schematic((end.x_cm, end.y_cm))
            next_route_index = min(len(route_points_cm) - 1, segment_index + 1)
            checkpoint_index = next(
                (checkpoint for checkpoint in checkpoints if checkpoint >= next_route_index),
                checkpoints[-1],
            )
            goal = route_points_cm[checkpoint_index]
            distance_to_goal = math.hypot(start.x_cm - goal.x_cm, start.y_cm - goal.y_cm)
            cv2.line(schematic, start_px, end_px, self.route_velocity_color(distance_to_goal), 3, cv2.LINE_AA)

    def robot_footprint_metric_polygons(
        self,
        pose: HybridPose,
        geometry: RobotGeometry,
    ) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        """Return base and intake rectangles in bottom-left field centimeters."""
        forward = (math.cos(pose.theta_rad), math.sin(pose.theta_rad))
        right = (math.sin(pose.theta_rad), -math.cos(pose.theta_rad))
        half_width_cm = geometry.width_cm * 0.5
        tube_half_width_cm = self.robot_config.tube_width_cm * 0.5

        front_center = (
            pose.x_cm + forward[0] * geometry.front_cm,
            pose.y_cm + forward[1] * geometry.front_cm,
        )
        rear_center = (
            pose.x_cm - forward[0] * geometry.rear_cm,
            pose.y_cm - forward[1] * geometry.rear_cm,
        )
        tube_front = (
            pose.x_cm + forward[0] * geometry.tube_forward_cm + right[0] * geometry.tube_right_cm,
            pose.y_cm + forward[1] * geometry.tube_forward_cm + right[1] * geometry.tube_right_cm,
        )
        tube_rear = (
            front_center[0] + right[0] * geometry.tube_right_cm,
            front_center[1] + right[1] * geometry.tube_right_cm,
        )

        base = [
            (front_center[0] + right[0] * half_width_cm, front_center[1] + right[1] * half_width_cm),
            (front_center[0] - right[0] * half_width_cm, front_center[1] - right[1] * half_width_cm),
            (rear_center[0] - right[0] * half_width_cm, rear_center[1] - right[1] * half_width_cm),
            (rear_center[0] + right[0] * half_width_cm, rear_center[1] + right[1] * half_width_cm),
        ]
        intake = [
            (tube_front[0] + right[0] * tube_half_width_cm, tube_front[1] + right[1] * tube_half_width_cm),
            (tube_front[0] - right[0] * tube_half_width_cm, tube_front[1] - right[1] * tube_half_width_cm),
            (tube_rear[0] - right[0] * tube_half_width_cm, tube_rear[1] - right[1] * tube_half_width_cm),
            (tube_rear[0] + right[0] * tube_half_width_cm, tube_rear[1] + right[1] * tube_half_width_cm),
        ]
        return base, intake

    def unload_mechanism_metric_polygon(
        self,
        pose: HybridPose,
        geometry: RobotGeometry,
    ) -> list[tuple[float, float]]:
        """Return the lowered rear unloading mechanism in bottom-left field centimeters."""
        if geometry.unload_extension_cm <= 0.0:
            return []
        forward = (math.cos(pose.theta_rad), math.sin(pose.theta_rad))
        right = (math.sin(pose.theta_rad), -math.cos(pose.theta_rad))
        half_width_cm = self.robot_config.tube_width_cm * 0.5
        rear_center = (
            pose.x_cm - forward[0] * geometry.rear_cm,
            pose.y_cm - forward[1] * geometry.rear_cm,
        )
        unload_tip = HybridAStarPlanner.rear_unload_point_for_pose(pose, geometry)
        return [
            (rear_center[0] + right[0] * half_width_cm, rear_center[1] + right[1] * half_width_cm),
            (rear_center[0] - right[0] * half_width_cm, rear_center[1] - right[1] * half_width_cm),
            (unload_tip[0] - right[0] * half_width_cm, unload_tip[1] - right[1] * half_width_cm),
            (unload_tip[0] + right[0] * half_width_cm, unload_tip[1] + right[1] * half_width_cm),
        ]

    def draw_field_goals(self, schematic: np.ndarray) -> None:
        """Draw the fixed goal openings centered on each field end."""
        center_y = self.field.height_cm * 0.5
        small_half_cm = 4.0
        large_half_cm = 10.0
        small_a = self.mapper.field_metric_cm_to_schematic((0.0, center_y - small_half_cm))
        small_b = self.mapper.field_metric_cm_to_schematic((0.0, center_y + small_half_cm))
        large_a = self.mapper.field_metric_cm_to_schematic((self.field.width_cm, center_y - large_half_cm))
        large_b = self.mapper.field_metric_cm_to_schematic((self.field.width_cm, center_y + large_half_cm))
        cv2.line(schematic, large_a, large_b, (180, 220, 180), 7, cv2.LINE_AA)
        cv2.line(schematic, small_a, small_b, (0, 255, 180), 7, cv2.LINE_AA)

    def draw_robot_footprint_snapshot(
        self,
        schematic: np.ndarray,
        pose: HybridPose,
        geometry: RobotGeometry,
        alpha: float,
        base_color: tuple[int, int, int],
        intake_color: tuple[int, int, int],
        thickness: int,
        show_unload: bool = False,
    ) -> None:
        """Draw one stylized robot footprint snapshot for route orientation QA."""
        base_cm, intake_cm = self.robot_footprint_metric_polygons(pose, geometry)
        unload_cm = self.unload_mechanism_metric_polygon(pose, geometry) if show_unload else []
        base_px = np.array([self.mapper.field_metric_cm_to_schematic(point) for point in base_cm], dtype=np.int32).reshape(-1, 1, 2)
        intake_px = np.array([self.mapper.field_metric_cm_to_schematic(point) for point in intake_cm], dtype=np.int32).reshape(-1, 1, 2)
        unload_px = (
            np.array([self.mapper.field_metric_cm_to_schematic(point) for point in unload_cm], dtype=np.int32).reshape(-1, 1, 2)
            if unload_cm
            else None
        )
        origin_px = self.mapper.field_metric_cm_to_schematic((pose.x_cm, pose.y_cm))
        tube_px = self.mapper.field_metric_cm_to_schematic(HybridAStarPlanner.tube_center_for_pose(pose, geometry))
        unload_tip_px = (
            self.mapper.field_metric_cm_to_schematic(HybridAStarPlanner.rear_unload_point_for_pose(pose, geometry))
            if show_unload
            else None
        )

        overlay = schematic.copy()
        cv2.fillPoly(overlay, [base_px], base_color, cv2.LINE_AA)
        cv2.fillPoly(overlay, [intake_px], intake_color, cv2.LINE_AA)
        if unload_px is not None:
            cv2.fillPoly(overlay, [unload_px], (0, 255, 180), cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, schematic, 1.0 - alpha, 0.0, schematic)
        cv2.polylines(schematic, [base_px], True, base_color, thickness, cv2.LINE_AA)
        cv2.polylines(schematic, [intake_px], True, intake_color, max(1, thickness - 1), cv2.LINE_AA)
        if unload_px is not None:
            cv2.polylines(schematic, [unload_px], True, (0, 255, 180), max(1, thickness - 1), cv2.LINE_AA)
        cv2.arrowedLine(schematic, origin_px, tube_px, intake_color, thickness, cv2.LINE_AA, tipLength=0.35)
        if unload_tip_px is not None:
            cv2.line(schematic, origin_px, unload_tip_px, (0, 255, 180), max(1, thickness - 1), cv2.LINE_AA)
            cv2.circle(schematic, unload_tip_px, max(3, thickness + 1), (0, 255, 180), -1, cv2.LINE_AA)
        cv2.circle(schematic, origin_px, max(3, thickness + 2), base_color, -1, cv2.LINE_AA)

    def draw_route_heading_indicators(
        self,
        schematic: np.ndarray,
        route: list[HybridPose],
        geometry: RobotGeometry,
        interval: int = 20,
    ) -> None:
        """Draw cyan arrows showing heading along the trajectory."""
        if not route:
            return
        for index in range(0, len(route), max(1, interval)):
            pose = route[index]
            origin_px = self.mapper.field_metric_cm_to_schematic((pose.x_cm, pose.y_cm))
            tube_px = self.mapper.field_metric_cm_to_schematic(HybridAStarPlanner.tube_center_for_pose(pose, geometry))
            cv2.arrowedLine(schematic, origin_px, tube_px, (255, 255, 0), 2, cv2.LINE_AA, tipLength=0.35)

    def draw_loupe(self, frame: np.ndarray, cursor: tuple[int, int]) -> np.ndarray:
        """Draw the zoomed cursor loupe used during manual corner selection."""
        overlay = frame.copy()
        height, width = overlay.shape[:2]
        crop_w = min(self.mapper.camera.loupe_crop_size, width)
        crop_h = min(self.mapper.camera.loupe_crop_size, height)

        x0, x1 = self._clamp_crop_bounds(cursor[0], crop_w, width)
        y0, y1 = self._clamp_crop_bounds(cursor[1], crop_h, height)
        crop = overlay[y0:y1, x0:x1]
        loupe = cv2.resize(
            crop,
            (crop.shape[1] * self.mapper.camera.loupe_scale, crop.shape[0] * self.mapper.camera.loupe_scale),
            interpolation=cv2.INTER_NEAREST,
        )

        loupe_h, loupe_w = loupe.shape[:2]
        dest_x1 = width - self.mapper.camera.loupe_padding
        dest_x0 = max(0, dest_x1 - loupe_w)
        dest_y0 = self.mapper.camera.loupe_padding
        dest_y1 = min(height, dest_y0 + loupe_h)

        visible_loupe = loupe[: dest_y1 - dest_y0, : dest_x1 - dest_x0]
        overlay[dest_y0:dest_y1, dest_x0:dest_x1] = visible_loupe
        cv2.rectangle(overlay, (dest_x0, dest_y0), (dest_x1, dest_y1), (255, 255, 255), 2)

        center_x = dest_x0 + visible_loupe.shape[1] // 2
        center_y = dest_y0 + visible_loupe.shape[0] // 2
        cv2.line(overlay, (center_x, dest_y0), (center_x, dest_y1), (0, 255, 255), 1)
        cv2.line(overlay, (dest_x0, center_y), (dest_x1, center_y), (0, 255, 255), 1)
        cv2.putText(
            overlay,
            "Loupe",
            (dest_x0, max(20, dest_y0 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return overlay

    @staticmethod
    def _clamp_crop_bounds(center: int, crop_size: int, limit: int) -> tuple[int, int]:
        """Clamp a crop region so the loupe stays inside frame bounds."""
        if limit <= crop_size:
            return 0, limit
        half = crop_size // 2
        start = max(0, center - half)
        end = start + crop_size
        if end > limit:
            end = limit
            start = end - crop_size
        return start, end

    def draw_detected_aruco_centers(
        self,
        frame: np.ndarray,
        marker_centers: dict[int, np.ndarray],
    ) -> np.ndarray:
        """Overlay the detected ArUco marker centers used for auto calibration."""
        overlay = frame.copy()
        for marker_id in self.mapper.camera.required_aruco_ids:
            if marker_id not in marker_centers:
                continue

            center = marker_centers[marker_id]
            point = (int(round(center[0])), int(round(center[1])))
            cv2.circle(overlay, point, 6, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.putText(
                overlay,
                f"ID {marker_id}",
                (point[0] + 10, point[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        return overlay

    def draw_projected_aruco_debug(
        self,
        frame: np.ndarray,
        transform_matrix: np.ndarray,
    ) -> np.ndarray:
        """Project the field and marker-center geometry back onto the selector view."""
        overlay = frame.copy()
        inverse_transform = np.linalg.inv(transform_matrix)

        field_outline = cv2.perspectiveTransform(
            self.mapper.destination_corners(self.mapper.camera.topdown_warp_size).reshape(-1, 1, 2),
            inverse_transform,
        ).astype(np.int32)

        total_offset_cm = self.mapper.camera.wall_thickness_cm + self.mapper.camera.marker_outer_offset_cm
        world_points_cm = (
            (-total_offset_cm, -total_offset_cm),
            (self.field.width_cm + total_offset_cm, -total_offset_cm),
            (self.field.width_cm + total_offset_cm, self.field.height_cm + total_offset_cm),
            (-total_offset_cm, self.field.height_cm + total_offset_cm),
        )
        marker_destination = np.array(
            [self.mapper.field_cm_to_topdown_px_unflipped(x_cm, y_cm) for x_cm, y_cm in world_points_cm],
            dtype=np.float32,
        )
        marker_outline = cv2.perspectiveTransform(
            marker_destination.reshape(-1, 1, 2),
            inverse_transform,
        ).astype(np.int32)

        cv2.polylines(overlay, [field_outline], True, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.polylines(overlay, [marker_outline], True, (255, 200, 0), 2, cv2.LINE_AA)

        field_labels = ("Field TL", "Field TR", "Field BR", "Field BL")
        for label, point in zip(field_labels, field_outline.reshape(-1, 2)):
            point_xy = (int(point[0]), int(point[1]))
            cv2.circle(overlay, point_xy, 5, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.putText(
                overlay,
                label,
                (point_xy[0] + 8, point_xy[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        marker_labels = ("Marker TL", "Marker TR", "Marker BR", "Marker BL")
        for label, point in zip(marker_labels, marker_outline.reshape(-1, 2)):
            point_xy = (int(point[0]), int(point[1]))
            cv2.circle(overlay, point_xy, 4, (255, 200, 0), -1, cv2.LINE_AA)
            cv2.putText(
                overlay,
                label,
                (point_xy[0] + 8, point_xy[1] + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 200, 0),
                2,
                cv2.LINE_AA,
            )

        cv2.putText(
            overlay,
            "Yellow: inferred inner field   Blue: expected ArUco center quad",
            (16, overlay.shape[0] - 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return overlay

    def draw_video_pause_overlay(self, frame: np.ndarray) -> None:
        """Draw the legacy paused-video status text."""
        cv2.putText(
            frame,
            "PAUSED - press Space or p to resume",
            (20, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def draw_pickup_footprints(
        self,
        schematic: np.ndarray,
        pickup_poses: list[HybridPose],
        geometry: RobotGeometry,
    ) -> None:
        """Draw bold footprints at every planned ball pickup pose."""
        for pickup_pose in pickup_poses:
            self.draw_robot_footprint_snapshot(
                schematic,
                pickup_pose,
                geometry,
                alpha=0.48,
                base_color=(255, 0, 255),
                intake_color=(0, 255, 255),
                thickness=3,
            )

    def draw_control_xte_on_schematic(
        self,
        schematic: np.ndarray,
        robot_pose: RobotPose | None,
        drive_runtime: DriveRuntime | None,
    ) -> None:
        """Draw the closest-route projection used by the local controller."""
        if drive_runtime is None or drive_runtime.last_error is None or robot_pose is None:
            return
        robot_px = self.mapper.field_metric_cm_to_schematic((robot_pose.x_cm, robot_pose.y_cm))
        closest_px = self.mapper.field_metric_cm_to_schematic(drive_runtime.last_error.closest_point_cm)
        cv2.line(schematic, robot_px, closest_px, (0, 165, 255), 3, cv2.LINE_AA)
        cv2.circle(schematic, closest_px, 6, (0, 165, 255), -1, cv2.LINE_AA)

    def draw_control_xte_on_topdown(
        self,
        frame: np.ndarray,
        robot_pose: RobotPose | None,
        drive_runtime: DriveRuntime | None,
    ) -> None:
        """Draw XTE as a robot-to-route line in the annotated top-down camera view."""
        if drive_runtime is None or drive_runtime.last_error is None or robot_pose is None:
            return
        robot_px = tuple(int(round(v)) for v in self.mapper.field_cm_to_topdown_pixel((robot_pose.x_cm, robot_pose.y_cm)))
        closest_px = tuple(int(round(v)) for v in self.mapper.field_cm_to_topdown_pixel(drive_runtime.last_error.closest_point_cm))
        cv2.line(frame, robot_px, closest_px, (0, 165, 255), 3, cv2.LINE_AA)
        cv2.circle(frame, closest_px, 6, (0, 165, 255), -1, cv2.LINE_AA)

    def draw_robot_marker_debug(
        self,
        frame: np.ndarray,
        observations: dict[int, RobotMarkerObservation],
        calibration: dict[str, Any] | None,
        robot_origin_px: tuple[float, float] | None,
        robot_pose: RobotPose | None,
        params: dict[str, object] | None = None,
        runtime: RobotCalibrationRuntime | None = None,
    ) -> None:
        """Draw robot ArUco, parallax projection, origin, footprint, and pickup tube."""
        if runtime is not None:
            colors = [(0, 180, 255), (255, 180, 0)]
            for index, marker_id in enumerate(self.robot_config.marker_ids):
                points = runtime.collected_points.get(marker_id, [])
                if len(points) >= 2:
                    pts = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
                    cv2.polylines(frame, [pts], False, colors[index % len(colors)], 2, cv2.LINE_AA)
                if marker_id in runtime.fitted_centers:
                    center = runtime.fitted_centers[marker_id]
                    cv2.drawMarker(
                        frame,
                        (int(round(center[0])), int(round(center[1]))),
                        (0, 165, 255),
                        cv2.MARKER_TILTED_CROSS,
                        24,
                        2,
                        cv2.LINE_AA,
                    )

        for observation in observations.values():
            raw_pts = observation.corners.astype(np.int32).reshape(-1, 1, 2)
            ground_pts = observation.ground_corners.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(frame, [raw_pts], True, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.polylines(frame, [ground_pts], True, (255, 255, 0), 1, cv2.LINE_AA)

            raw_center = tuple(int(round(v)) for v in observation.center)
            ground_center = tuple(int(round(v)) for v in observation.ground_center)
            cv2.circle(frame, raw_center, 4, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.drawMarker(frame, ground_center, (255, 255, 0), cv2.MARKER_CROSS, 16, 2, cv2.LINE_AA)
            cv2.line(frame, raw_center, ground_center, (255, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(
                frame,
                f"Robot ID {observation.marker_id}",
                (raw_center[0] + 8, raw_center[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        if robot_origin_px is None:
            return

        origin = np.array(robot_origin_px, dtype=np.float32)
        origin_i = (int(round(origin[0])), int(round(origin[1])))
        cv2.circle(frame, origin_i, 8, (255, 90, 30), -1, cv2.LINE_AA)
        cv2.circle(frame, origin_i, 15, (255, 255, 255), 2, cv2.LINE_AA)

        for observation in observations.values():
            if calibration is None or str(observation.marker_id) not in calibration.get("markers", {}):
                continue
            ground_center = tuple(int(round(v)) for v in observation.ground_center)
            cv2.line(frame, ground_center, origin_i, (255, 90, 30), 2, cv2.LINE_AA)

        if robot_pose is None:
            return

        source_height, source_width = frame.shape[:2]
        px_per_cm_x = (source_width - 1) / self.field.width_cm
        px_per_cm_y = (source_height - 1) / self.field.height_cm

        def field_delta_to_px(dx_cm: float, dy_cm: float) -> np.ndarray:
            return np.array([dx_cm * px_per_cm_x, -dy_cm * px_per_cm_y], dtype=np.float32)

        forward = (math.cos(robot_pose.heading_rad), math.sin(robot_pose.heading_rad))
        right = (math.sin(robot_pose.heading_rad), -math.cos(robot_pose.heading_rad))
        geometry = self.robot_geometry_from_params(params)
        half_width_cm = geometry.width_cm * 0.5
        front_center = origin + field_delta_to_px(
            forward[0] * geometry.front_cm,
            forward[1] * geometry.front_cm,
        )
        rear_center = origin + field_delta_to_px(
            -forward[0] * geometry.rear_cm,
            -forward[1] * geometry.rear_cm,
        )
        right_px = field_delta_to_px(right[0] * half_width_cm, right[1] * half_width_cm)
        footprint = np.array(
            [
                front_center + right_px,
                front_center - right_px,
                rear_center - right_px,
                rear_center + right_px,
            ],
            dtype=np.int32,
        ).reshape(-1, 1, 2)
        cv2.polylines(frame, [footprint], True, (255, 90, 30), 2, cv2.LINE_AA)

        tube_px = origin + field_delta_to_px(
            forward[0] * geometry.tube_forward_cm + right[0] * geometry.tube_right_cm,
            forward[1] * geometry.tube_forward_cm + right[1] * geometry.tube_right_cm,
        )
        tube_i = tuple(int(round(v)) for v in tube_px)
        cv2.arrowedLine(frame, origin_i, tube_i, (255, 255, 255), 2, cv2.LINE_AA, tipLength=0.25)
        cv2.circle(frame, tube_i, 10, (0, 255, 255), 3, cv2.LINE_AA)
        cv2.circle(frame, tube_i, 3, (0, 255, 255), -1, cv2.LINE_AA)

    def draw_robot_calibration_status(
        self,
        frame: np.ndarray,
        runtime: RobotCalibrationRuntime,
        robot_pose: RobotPose | None,
        params: dict[str, object] | None = None,
    ) -> None:
        """Draw the legacy yellow robot calibration, shortcut, geometry, and pose text."""
        counts = ", ".join(
            f"ID {marker_id}: {len(runtime.collected_points.get(marker_id, []))}"
            for marker_id in self.robot_config.marker_ids
        )
        if runtime.phase == RobotCalibrationPhase.STATE_CALIBRATING_SPIN:
            mode = "ROBOT CAL: SPIN"
            action = "Spin robot, press s to fit"
        elif runtime.phase == RobotCalibrationPhase.STATE_CALIBRATING_FORWARD:
            mode = "ROBOT CAL: FORWARD"
            action = "Face robot forward, press Enter"
        else:
            mode = "ROBOT TRACKING"
            action = "c: calibrate spin | w: save heading"

        geometry = self.robot_geometry_from_params(params)
        lines = [
            mode,
            action,
            f"Spin points: {counts}",
            (
                f"Geom W:{geometry.width_cm:.1f} F/R:{geometry.front_cm:.1f}/{geometry.rear_cm:.1f} "
                f"Tube:{geometry.tube_forward_cm:.1f},{geometry.tube_right_cm:.1f} "
                f"Unload:{geometry.unload_extension_cm:.1f}"
            ),
        ]
        if robot_pose is not None:
            lines.append(
                f"Robot: X={robot_pose.x_cm:.1f}cm Y={robot_pose.y_cm:.1f}cm H={math.degrees(robot_pose.heading_rad):.1f}deg"
            )
        elif runtime.calibration is None:
            lines.append("No robot_calibration.json loaded")
        else:
            lines.append("Robot marker not detected")
        if runtime.warning:
            lines.append(runtime.warning)

        y0 = 62
        for index, line in enumerate(lines):
            cv2.putText(
                frame,
                line,
                (20, y0 + index * 23),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.56,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

    def draw_drive_status(self, frame: np.ndarray, drive_runtime: DriveRuntime | None) -> None:
        """Draw live XTE, heading error, and wheel dispatch state."""
        if drive_runtime is None:
            return
        command = drive_runtime.last_command
        lines = [
            f"State: {drive_runtime.state.value}",
            f"Motor: L={command.left_pct:.0f} R={command.right_pct:.0f}",
        ]
        if drive_runtime.last_error is not None:
            lines.append(
                f"XTE: {drive_runtime.last_error.xte_cm:.1f} cm  "
                f"Head err: {math.degrees(drive_runtime.last_error.heading_error_rad):.1f} deg"
            )
        if drive_runtime.last_message:
            lines.append(drive_runtime.last_message[:70])

        y0 = frame.shape[0] - 112
        for index, line in enumerate(lines):
            cv2.putText(
                frame,
                line,
                (20, max(24, y0 + index * 24)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

    def annotate_camera_frame(
        self,
        frame_bgr: np.ndarray,
        red_zones: list[RedZoneDetection],
        white_balls: list[BallDetection],
        orange_balls: list[BallDetection],
        fps: float,
    ) -> np.ndarray:
        """Draw detections and lightweight debug text on the camera image."""
        annotated = frame_bgr.copy()
        for zone in red_zones:
            x, y, width, height = zone.bounding_box
            cv2.drawContours(annotated, [zone.contour], -1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.rectangle(annotated, (x, y), (x + width, y + height), (0, 80, 255), 1, cv2.LINE_AA)
            cv2.putText(
                annotated,
                f"red {int(zone.area)}",
                (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
        for ball in white_balls:
            cv2.circle(annotated, ball.center, ball.radius_px, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(annotated, ball.center, 2, (200, 200, 200), -1, cv2.LINE_AA)
            cv2.putText(annotated, f"W c={ball.circularity:.2f}", (ball.center[0] + 10, ball.center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
        for orange_ball in orange_balls:
            cv2.circle(annotated, orange_ball.center, orange_ball.radius_px, (0, 140, 255), 2, cv2.LINE_AA)
            cv2.circle(annotated, orange_ball.center, 2, (0, 180, 255), -1, cv2.LINE_AA)
            cv2.putText(annotated, f"O c={orange_ball.circularity:.2f}", (orange_ball.center[0] + 10, orange_ball.center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 140, 255), 1, cv2.LINE_AA)
        cv2.putText(annotated, f"FPS: {fps:.1f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        return annotated

    def build_mask_preview(self, red_mask: np.ndarray, white_mask: np.ndarray, orange_mask: np.ndarray) -> np.ndarray:
        """Create a compact debug view for the segmentation stage."""
        red_bgr = cv2.cvtColor(red_mask, cv2.COLOR_GRAY2BGR)
        white_bgr = cv2.cvtColor(white_mask, cv2.COLOR_GRAY2BGR)
        orange_bgr = cv2.cvtColor(orange_mask, cv2.COLOR_GRAY2BGR)
        cv2.putText(red_bgr, "Red", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(white_bgr, "White YOLO", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(orange_bgr, "Orange YOLO", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 140, 255), 2, cv2.LINE_AA)
        return np.hstack((red_bgr, white_bgr, orange_bgr))

    @staticmethod
    def resize_to_match_height(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Resize the right panel so both panels can be stacked horizontally."""
        target_height = left.shape[0]
        if right.shape[0] == target_height:
            return left, right
        new_width = int(right.shape[1] * target_height / max(1, right.shape[0]))
        resized = cv2.resize(right, (new_width, target_height), interpolation=cv2.INTER_LINEAR)
        return left, resized

    def make_topdown_placeholder(self, message: str) -> np.ndarray:
        """Create a deterministic placeholder while the manual warp is not ready."""
        width, height = self.mapper.camera.topdown_warp_size
        placeholder = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(placeholder, message, (30, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        return placeholder

    def draw_schematic(
        self,
        frame_shape: tuple[int, int, int],
        red_zones: list[RedZoneDetection],
        smoothed_ball_coordinates: list[SmoothedBallCoordinate],
        camera_center_pixels: tuple[float, float],
        route_points_cm: list[HybridPose] | None = None,
        route_pickup_poses_cm: list[HybridPose] | None = None,
        route_unload_pose_cm: HybridPose | None = None,
        route_unload_goal_cm: tuple[float, float] | None = None,
        selected_start_cm: tuple[int, int] | None = None,
        selected_ball_track_id: int | None = None,
        robot_pose: RobotPose | None = None,
        params: dict[str, object] | None = None,
        drive_runtime: DriveRuntime | None = None,
        num_intermediate_snapshots: int = 0,
        route_heading_marker_interval: int = 20,
    ) -> np.ndarray:
        """Draw a clean synthetic field view containing detected objects and route state."""
        source_height, source_width = frame_shape[:2]
        schematic = np.full(
            (self.window.schematic_height_px, self.window.schematic_width_px, 3),
            (40, 100, 40),
            dtype=np.uint8,
        )
        cv2.rectangle(
            schematic,
            (0, 0),
            (self.window.schematic_width_px - 1, self.window.schematic_height_px - 1),
            (220, 220, 220),
            2,
            cv2.LINE_AA,
        )
        self.draw_field_goals(schematic)

        for zone in red_zones:
            mapped_contour = zone.corrected_contour.astype(np.float32).copy()
            mapped_contour[:, 0, 0] *= self.window.schematic_width_px / max(1, source_width)
            mapped_contour[:, 0, 1] *= self.window.schematic_height_px / max(1, source_height)
            cv2.polylines(schematic, [mapped_contour.astype(np.int32)], True, (0, 0, 255), 3, cv2.LINE_AA)

        for ball in smoothed_ball_coordinates:
            center = self.mapper.field_metric_cm_to_schematic((ball.cm_x, ball.cm_y))
            radius = max(4, int(ball.radius_px * self.window.schematic_width_px / max(1, source_width)))
            fill_color, edge_color = ((245, 245, 245), (120, 120, 120)) if ball.label == "white" else ((0, 140, 255), (0, 80, 180))
            cv2.circle(schematic, center, radius, fill_color, -1, cv2.LINE_AA)
            cv2.circle(schematic, center, radius, edge_color, 1, cv2.LINE_AA)

            label = f"X: {ball.cm_x:.1f}, Y: {ball.cm_y:.1f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            text_x = center[0] + 10
            if text_x + text_width > self.window.schematic_width_px - 1:
                text_x = max(0, center[0] - 10 - text_width)
            text_y = center[1] - 10
            min_text_y = text_height + baseline
            if text_y < min_text_y:
                text_y = min(self.window.schematic_height_px - baseline - 1, center[1] + text_height + 10)
            cv2.putText(schematic, label, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        geometry = self.robot_geometry_from_params(params)
        if route_points_cm:
            self.draw_velocity_profile_route(schematic, route_points_cm, route_pickup_poses_cm)
            self.draw_route_heading_indicators(
                schematic,
                route_points_cm,
                geometry,
                interval=route_heading_marker_interval,
            )
            if num_intermediate_snapshots > 0 and len(route_points_cm) > 2:
                denominator = num_intermediate_snapshots + 1
                for index in sorted(
                    {
                        int(round((len(route_points_cm) - 1) * sample_index / denominator))
                        for sample_index in range(1, num_intermediate_snapshots + 1)
                    }
                ):
                    self.draw_robot_footprint_snapshot(schematic, route_points_cm[index], geometry, 0.18, (255, 0, 255), (255, 255, 0), 1)
            self.draw_pickup_footprints(schematic, route_pickup_poses_cm or [], geometry)
            if route_unload_pose_cm is not None:
                self.draw_robot_footprint_snapshot(
                    schematic,
                    route_unload_pose_cm,
                    geometry,
                    0.28,
                    (255, 0, 255),
                    (255, 255, 0),
                    2,
                    show_unload=True,
                )
            if route_unload_goal_cm is not None:
                goal_px = self.mapper.field_metric_cm_to_schematic(route_unload_goal_cm)
                cv2.drawMarker(schematic, goal_px, (0, 255, 180), cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)

        if selected_start_cm is not None:
            if robot_pose is not None:
                selected_start = self.mapper.field_metric_cm_to_schematic((robot_pose.x_cm, robot_pose.y_cm))
            else:
                selected_ball = next(
                    (ball for ball in smoothed_ball_coordinates if ball.track_id == selected_ball_track_id),
                    None,
                )
                selected_start = (
                    self.mapper.field_metric_cm_to_schematic((selected_ball.cm_x, selected_ball.cm_y))
                    if selected_ball is not None
                    else self.mapper.field_cm_to_schematic(selected_start_cm)
                )
            cv2.circle(schematic, selected_start, 8, (0, 255, 255), 2, cv2.LINE_AA)

        if robot_pose is not None:
            pose = robot_pose
            robot_center = self.mapper.field_metric_cm_to_schematic((pose.x_cm, pose.y_cm))
            live_pose = HybridPose(pose.x_cm, pose.y_cm, pose.heading_rad)
            base_cm, intake_cm = self.robot_footprint_metric_polygons(live_pose, geometry)
            footprint_px = np.array([self.mapper.field_metric_cm_to_schematic(point) for point in base_cm], dtype=np.int32).reshape(-1, 1, 2)
            intake_px = np.array([self.mapper.field_metric_cm_to_schematic(point) for point in intake_cm], dtype=np.int32).reshape(-1, 1, 2)
            tube_center = self.mapper.field_metric_cm_to_schematic((pose.tube_x_cm, pose.tube_y_cm))
            cv2.polylines(schematic, [footprint_px], True, (255, 90, 30), 2, cv2.LINE_AA)
            cv2.polylines(schematic, [intake_px], True, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(schematic, robot_center, 7, (255, 90, 30), -1, cv2.LINE_AA)
            cv2.circle(schematic, robot_center, 11, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.arrowedLine(schematic, robot_center, tube_center, (255, 255, 255), 2, cv2.LINE_AA, tipLength=0.28)
            cv2.circle(schematic, tube_center, 10, (0, 255, 255), 3, cv2.LINE_AA)
            cv2.circle(schematic, tube_center, 3, (0, 255, 255), -1, cv2.LINE_AA)
            robot_label = f"Robot X:{pose.x_cm:.1f} Y:{pose.y_cm:.1f} Tube:{pose.tube_x_cm:.1f},{pose.tube_y_cm:.1f}"
            cv2.putText(
                schematic,
                robot_label,
                (max(10, min(robot_center[0] + 16, self.window.schematic_width_px - 360)), max(25, robot_center[1] - 16)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        self.draw_control_xte_on_schematic(schematic, robot_pose, drive_runtime)
        camera_center_schematic = self.mapper.map_point_between_frames(
            (int(round(camera_center_pixels[0])), int(round(camera_center_pixels[1]))),
            (source_width, source_height),
            self.window.schematic_size_px,
        )
        cv2.line(
            schematic,
            (camera_center_schematic[0] - 12, camera_center_schematic[1]),
            (camera_center_schematic[0] + 12, camera_center_schematic[1]),
            (255, 80, 80),
            3,
            cv2.LINE_AA,
        )
        cv2.line(
            schematic,
            (camera_center_schematic[0], camera_center_schematic[1] - 12),
            (camera_center_schematic[0], camera_center_schematic[1] + 12),
            (255, 80, 80),
            3,
            cv2.LINE_AA,
        )
        cv2.circle(schematic, camera_center_schematic, 7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(
            schematic,
            f"Field {self.field.width_cm:.1f}x{self.field.height_cm:.1f} cm",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            schematic,
            "White balls: "
            f"{sum(ball.label == 'white' for ball in smoothed_ball_coordinates)}  Orange balls: "
            f"{sum(ball.label == 'orange' for ball in smoothed_ball_coordinates)}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return schematic


SchematicRenderer = DebugRenderer
