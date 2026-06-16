"""Typed configuration for the top-down detector.

The legacy detector still owns the executable OpenCV loop during the refactor,
but these immutable config objects provide a single place for field geometry,
camera/UI defaults, detection tuning, robot geometry, planning, and control
values.  They intentionally mirror the existing constants so migration can be
done incrementally without changing behavior.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path


@dataclass(frozen=True)
class FieldConfig:
    """Measured contest field dimensions and logical grid sizing."""

    width_cm: float = 167.0
    height_cm: float = 121.5
    ball_height_cm: float = 2.0
    floor_height_cm: float = 0.0

    @property
    def grid_width_cm(self) -> int:
        return int(round(self.width_cm))

    @property
    def grid_height_cm(self) -> int:
        return int(round(self.height_cm))


@dataclass(frozen=True)
class PathConfig:
    """Repository-relative paths used by the detector application."""

    repo_root: Path
    calibration_file: Path | None = None
    robot_calibration_file: Path | None = None
    default_image: Path | None = None
    default_video_dir: Path | None = None
    yolo_model_path: Path | None = None

    def __post_init__(self) -> None:
        root = self.repo_root
        object.__setattr__(self, "calibration_file", self.calibration_file or root / "calibration_data.npz")
        object.__setattr__(self, "robot_calibration_file", self.robot_calibration_file or root / "robot_calibration.json")
        object.__setattr__(
            self,
            "default_image",
            self.default_image or root / "Bane_undistorted_transformed_close_balls.png",
        )
        object.__setattr__(self, "default_video_dir", self.default_video_dir or root / "videos")
        object.__setattr__(self, "yolo_model_path", self.yolo_model_path or Path("best.pt"))


@dataclass(frozen=True)
class CameraConfig:
    """Camera source and top-down calibration defaults."""

    use_live_feed: bool = False
    camera_index: int = 0
    topdown_warp_size: tuple[int, int] = (800, 600)
    required_aruco_ids: tuple[int, ...] = (0, 1, 2, 3)
    wall_thickness_cm: float = 1.6
    marker_outer_offset_cm: float = 8.0
    loupe_crop_size: int = 40
    loupe_scale: int = 5
    loupe_padding: int = 12
    point_radius: int = 6
    lk_fb_max_error_px: float = 2.0
    lk_ema_alpha: float = 0.2
    lk_win_size: tuple[int, int] = (31, 31)
    lk_max_level: int = 3
    lk_criteria_count: int = 30
    lk_criteria_epsilon: float = 0.01
    pose_loss_grace_frames: int = 3
    pose_loss_clear_route_frames: int = 15
    lock_focus_after_ball_count: bool = True
    camera_autofocus_enabled_during_prep: bool = True
    manual_focus_value: float = 0.0

    @property
    def lk_params(self) -> dict[str, object]:
        return {
            "winSize": self.lk_win_size,
            "maxLevel": self.lk_max_level,
            "criteria": (3, self.lk_criteria_count, self.lk_criteria_epsilon),
        }


@dataclass(frozen=True)
class WindowConfig:
    """OpenCV window names and fixed UI dimensions."""

    main_window_name: str = "Top-Down Detector"
    mask_window_name: str = "Segmentation Masks"
    control_color_window_name: str = "HSV Controls - Colors"
    control_filter_window_name: str = "HSV Controls - Filters"
    control_geometry_window_name: str = "HSV Controls - Geometry"
    manual_selector_window_name: str = "Manual Top-Down Selector"
    schematic_window_name: str = "2D Schematic"
    control_window_size: tuple[int, int] = (420, 520)
    schematic_size_px: tuple[int, int] = (900, 600)

    @property
    def schematic_width_px(self) -> int:
        return self.schematic_size_px[0]

    @property
    def schematic_height_px(self) -> int:
        return self.schematic_size_px[1]


@dataclass(frozen=True)
class RobotGeometryConfig:
    """Physical and tuned robot geometry in centimeters."""

    radius_cm: int = 15
    marker_ids: tuple[int, ...] = (4, 5)
    marker_height_cm: float = 9.0
    axle_distance_cm: float = 13.0
    track_width_cm: float = 20.0
    front_edge_from_front_axle_cm: float = 6.5
    tube_from_front_axle_cm: float = 10.5
    forward_heading_offset_rad: float = math.pi
    tube_right_offset_cm: float = 0.0
    tuned_footprint_width_cm: float = 20.0
    tuned_footprint_front_from_origin_cm: float = 8.3
    tuned_footprint_rear_from_origin_cm: float = 10.1
    tuned_tube_offset_cm: float = 17.1
    tuned_tube_right_offset_cm: float = 0.0
    tube_width_cm: float = 6.0
    tuned_unload_extension_cm: float = 30.0

    @property
    def front_axle_from_origin_cm(self) -> float:
        return self.axle_distance_cm * 0.5

    @property
    def front_edge_from_origin_cm(self) -> float:
        return self.front_axle_from_origin_cm + self.front_edge_from_front_axle_cm

    @property
    def rear_axle_from_origin_cm(self) -> float:
        return self.axle_distance_cm * 0.5

    @property
    def tube_offset_cm(self) -> float:
        return self.front_axle_from_origin_cm + self.tube_from_front_axle_cm

    @property
    def footprint_front_from_origin_cm(self) -> float:
        return self.front_axle_from_origin_cm

    @property
    def footprint_rear_from_origin_cm(self) -> float:
        return self.rear_axle_from_origin_cm

    @property
    def footprint_length_cm(self) -> float:
        return self.footprint_front_from_origin_cm + self.footprint_rear_from_origin_cm

    @property
    def footprint_width_cm(self) -> float:
        return self.track_width_cm


@dataclass(frozen=True)
class DetectionConfig:
    """Default trackbar values for segmentation, detection, and geometry tuning."""

    red1_h_min: int = 0
    red1_h_max: int = 12
    red2_h_min: int = 165
    red2_h_max: int = 179
    red_s_min: int = 155
    red_s_max: int = 255
    red_v_min: int = 174
    red_v_max: int = 255
    red_min_area: int = 400
    yolo_conf_pct: int = 50
    yolo_min_area: int = 157
    yolo_max_area: int = 1580
    cam_height_cm: int = 184
    calib_z_cm: int = 7
    heading_tuning: int = 180

    def trackbar_defaults(self, field: FieldConfig, robot: RobotGeometryConfig) -> dict[str, int]:
        return {
            "red1_h_min": self.red1_h_min,
            "red1_h_max": self.red1_h_max,
            "red2_h_min": self.red2_h_min,
            "red2_h_max": self.red2_h_max,
            "red_s_min": self.red_s_min,
            "red_s_max": self.red_s_max,
            "red_v_min": self.red_v_min,
            "red_v_max": self.red_v_max,
            "red_min_area": self.red_min_area,
            "yolo_conf_pct": self.yolo_conf_pct,
            "yolo_min_area": self.yolo_min_area,
            "yolo_max_area": self.yolo_max_area,
            "cam_height_cm": self.cam_height_cm,
            "calib_z_cm": self.calib_z_cm,
            "cam_center_x": int(round(field.width_cm * 0.5)),
            "cam_center_y": int(round(field.height_cm * 0.5)),
            "heading_tuning": self.heading_tuning,
            "robot_width_cmx10": int(round(robot.tuned_footprint_width_cm * 10.0)),
            "robot_front_cmx10": int(round(robot.tuned_footprint_front_from_origin_cm * 10.0)),
            "robot_rear_cmx10": int(round(robot.tuned_footprint_rear_from_origin_cm * 10.0)),
            "tube_forward_cmx10": int(round(robot.tuned_tube_offset_cm * 10.0)),
            "tube_right_cmx10": int(round((robot.tuned_tube_right_offset_cm + 50.0) * 10.0)),
            "unload_extension_cmx10": int(round(robot.tuned_unload_extension_cm * 10.0)),
        }


@dataclass(frozen=True)
class TrackbarConfig:
    """Names and window ownership for OpenCV trackbars."""

    names: dict[str, str] = dataclass_field(
        default_factory=lambda: {
            "red1_h_min": "R1 H min",
            "red1_h_max": "R1 H max",
            "red2_h_min": "R2 H min",
            "red2_h_max": "R2 H max",
            "red_s_min": "R S min",
            "red_s_max": "R S max",
            "red_v_min": "R V min",
            "red_v_max": "R V max",
            "red_min_area": "R min area",
            "yolo_conf_pct": "YOLO conf %",
            "yolo_min_area": "min area",
            "yolo_max_area": "max area",
            "cam_height_cm": "Cam h cm",
            "calib_z_cm": "Border h cm",
            "cam_center_x": "Cam X cm",
            "cam_center_y": "Cam Y cm",
            "heading_tuning": "Heading Tuning",
            "robot_width_cmx10": "Robot W x10",
            "robot_front_cmx10": "Body F x10",
            "robot_rear_cmx10": "Body R x10",
            "tube_forward_cmx10": "Tube F x10",
            "tube_right_cmx10": "Tube R+50 x10",
            "unload_extension_cmx10": "Unload ext x10",
        }
    )

    def windows(self, window: WindowConfig) -> dict[str, str]:
        return {
            "red1_h_min": window.control_color_window_name,
            "red1_h_max": window.control_color_window_name,
            "red2_h_min": window.control_color_window_name,
            "red2_h_max": window.control_color_window_name,
            "red_s_min": window.control_color_window_name,
            "red_s_max": window.control_color_window_name,
            "red_v_min": window.control_color_window_name,
            "red_v_max": window.control_color_window_name,
            "red_min_area": window.control_filter_window_name,
            "yolo_conf_pct": window.control_filter_window_name,
            "yolo_min_area": window.control_filter_window_name,
            "yolo_max_area": window.control_filter_window_name,
            "cam_height_cm": window.control_geometry_window_name,
            "calib_z_cm": window.control_geometry_window_name,
            "cam_center_x": window.control_geometry_window_name,
            "cam_center_y": window.control_geometry_window_name,
            "heading_tuning": window.control_geometry_window_name,
            "robot_width_cmx10": window.control_geometry_window_name,
            "robot_front_cmx10": window.control_geometry_window_name,
            "robot_rear_cmx10": window.control_geometry_window_name,
            "tube_forward_cmx10": window.control_geometry_window_name,
            "tube_right_cmx10": window.control_geometry_window_name,
            "unload_extension_cmx10": window.control_geometry_window_name,
        }


@dataclass(frozen=True)
class PlannerConfig:
    """Route-planning and route-cache tuning."""

    theta_bins: int = 36
    step_cm: float = 4.0
    goal_tolerance_cm: float = 4.0
    max_expansions: int = 36000
    translation_directions: tuple[float, ...] = (1.0, -1.0)
    reverse_cost_multiplier: float = 2.5
    rotation_deltas_rad: tuple[float, ...] = (math.radians(-10.0), math.radians(10.0))
    in_place_rotation_cost: float = 2.0
    heuristic_weight: float = 1.5
    gear_shift_penalty: float = 50.0
    steering_change_penalty: float = 3.0
    transit_speed_pct: float = 38.0
    pivot_speed_pct: float = 30.0
    creep_speed_pct: float = 7.0
    flexible_standoff_max_cm: float = 15.0
    flexible_standoff_min_cm: float = 0.0
    flexible_standoff_heading_tolerance_rad: float = math.radians(10.0)
    unload_staging_margin_cm: float = 2.0
    wall_pickup_prefer_distance_cm: float = 12.0
    wall_pickup_perpendicular_tolerance_rad: float = math.radians(35.0)
    num_intermediate_snapshots: int = 0
    route_heading_marker_interval: int = 20
    route_target_move_invalidate_cm: float = 5.0
    route_crosstrack_invalidate_cm: float = 14.0
    avoid_non_target_balls_enabled: bool = True
    ball_radius_cm: float = 2.0
    non_target_ball_extra_clearance_cm: float = 0.0
    ball_core_cost: float = 1000.0
    ball_close_cost: float = 200.0
    ball_warning_cost: float = 50.0
    ball_close_clearance_cm: float = 5.0
    ball_warning_clearance_cm: float = 10.0

    @property
    def route_target_reached_cm(self) -> float:
        return self.goal_tolerance_cm


@dataclass(frozen=True)
class DriveConfig:
    """Integrated drive-control and TCP dispatch tuning."""

    robot_ip: str = "172.20.10.8"
    robot_tcp_port: int = 5555
    robot_command_format: str = "LR {left:.1f} {right:.1f}"
    """IMPORTANT VALUES"""
    drive_speed_pct: float = 100.0
    max_heading_for_forward_rad: float = math.radians(10.0)
    turn_speed_pct: float = 10.0

    max_cross_track_error_cm: float = 8.0
    max_speed_pct: float = 100.0
    heading_kp: float = 38.0
    xte_kp: float = 2.2
    heading_kd: float = 6.0
    xte_kd: float = 0.25
    cruise_distance_cm: float = 30.0
    creep_distance_cm: float = 10.0
    creep_speed_pct: float = 7.0
    near_zone_cm: float = 15.0
    near_zone_turn_speed_pct: float = 30.0
    near_zone_move_speed_pct: float = 7.0
    visual_servo_noise_floor_cm: float = 0.25
    visual_servo_min_improvement_cm: float = 0.08
    visual_servo_stall_frames: int = 3
    visual_servo_max_iterations: int = 12
    visual_servo_turn_kp: float = 0.85
    visual_servo_min_turn_deg: float = 0.25
    visual_servo_max_turn_deg: float = 8.0
    visual_servo_min_turn_speed_pct: float = 8.0
    visual_servo_settle_time_s: float = 0.5
    edge_slowdown_cm: float = 15.0
    edge_min_speed_scale: float = 0.35
    edge_max_gain_scale: float = 1.6
    # Percent-speed-per-second kinematic limits used by both runtime control
    # and the route heatmap. Tune these with robot battery/load conditions.
    acceleration_limit_pct_per_s: float = 45.0
    deceleration_limit_pct_per_s: float = 90.0
    blind_approach_trigger_cm: float = 8.0
    blind_approach_duration_s: float = 1.0
    min_send_interval_s: float = 0.02
    command_deadband_pct: float = 1.0
    manual_move_units: int = 5
    manual_move_speed: int = 40
    manual_turn_degrees: int = 15
    manual_turn_speed: int = 30
    drive_calibration_turn_degrees: float = 360.0
    drive_calibration_move_cm: float = 10.0
    drive_calibration_turn_speed_pct: float = 20.0
    drive_calibration_move_speed_pct: float = 15.0
    drive_calibration_settle_time_s: float = 0.5
    drive_calibration_min_actual_turn_deg: float = 45.0
    drive_calibration_min_actual_distance_cm: float = 1.0
    drive_calibration_max_origin_disagreement_cm: float = 3.0
    post_pickup_escape_clearance_cm: float = 4.0
    post_pickup_align_clearance_cm: float = 12.0
    post_pickup_escape_back_cm: float = 8.0
    post_pickup_escape_speed_pct: float = 8.0
    post_pickup_align_tolerance_deg: float = 15.0
    post_pickup_align_speed_pct: float = 18.0
    route_tracking_lookahead_segments: int = 12
    unload_staging_distance_cm: float = 10.0
    unload_pivot_tolerance_deg: float = 12.0
    unload_pivot_speed_pct: float = 18.0
    unload_trigger_distance_cm: float = 8.0
    unload_pipe_shake_units: float = 2.0
    unload_pipe_shake_speed: int = 35
    unload_pipe_shake_cycles: int = 1
    pivot_intercept_cm: float = 8.0
    pivot_turn_speed_pct: float = 20.0
    pivot_settle_time_s: float = 0.5
    pivot_heading_tolerance_deg: float = 5.0
    pivot_max_correction_attempts: int = 2
    pivot_min_turn_deg: float = 1.0
    key_left_arrow: set[int] = dataclass_field(default_factory=lambda: {2424832, 65361, 63234, 81})
    key_up_arrow: set[int] = dataclass_field(default_factory=lambda: {2490368, 65362, 63232, 82})
    key_right_arrow: set[int] = dataclass_field(default_factory=lambda: {2555904, 65363, 63235, 83})
    key_down_arrow: set[int] = dataclass_field(default_factory=lambda: {2621440, 65364, 63233, 84})
    telemetry_ringbuffer_size: int = 300
    telemetry_auto_dump_enabled: bool = True
    telemetry_dump_dir: str = "telemetry"
    # Turn speed profile
    turn_creep_speed_pct: float = 8.0
    turn_cruise_angle_deg: float = 30.0
    turn_creep_angle_deg: float = 8.0
    # Adjust gain (degrees of heading error -> speed% differential)
    adjust_gain: float = 2.0


@dataclass(frozen=True)
class RobotCalibrationConfig:
    """Robot calibration collection thresholds."""

    min_robot_spin_points: int = 20
    ellipse_warning_ratio: float = 1.12


@dataclass(frozen=True)
class AppConfig:
    """Top-level configuration bundle for the detector application."""

    paths: PathConfig
    field: FieldConfig = dataclass_field(default_factory=FieldConfig)
    camera: CameraConfig = dataclass_field(default_factory=CameraConfig)
    windows: WindowConfig = dataclass_field(default_factory=WindowConfig)
    robot: RobotGeometryConfig = dataclass_field(default_factory=RobotGeometryConfig)
    detection: DetectionConfig = dataclass_field(default_factory=DetectionConfig)
    trackbars: TrackbarConfig = dataclass_field(default_factory=TrackbarConfig)
    planner: PlannerConfig = dataclass_field(default_factory=PlannerConfig)
    drive: DriveConfig = dataclass_field(default_factory=DriveConfig)
    robot_calibration: RobotCalibrationConfig = dataclass_field(default_factory=RobotCalibrationConfig)

    @classmethod
    def from_repo_root(cls, repo_root: Path) -> "AppConfig":
        return cls(paths=PathConfig(repo_root=repo_root))
