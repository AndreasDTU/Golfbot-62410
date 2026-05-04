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
    tuned_unload_extension_cm: float = 15.0

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
    cam_height_cm: int = 179
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
    reverse_cost_multiplier: float = 1.8
    rotation_deltas_rad: tuple[float, ...] = (math.radians(-10.0), math.radians(10.0))
    in_place_rotation_cost: float = 1.1
    num_intermediate_snapshots: int = 0
    route_heading_marker_interval: int = 20
    route_target_move_invalidate_cm: float = 5.0
    route_crosstrack_invalidate_cm: float = 14.0

    @property
    def route_target_reached_cm(self) -> float:
        return self.goal_tolerance_cm


@dataclass(frozen=True)
class DriveConfig:
    """Integrated drive-control and UDP dispatch tuning."""

    robot_ip: str = "192.168.1.42"
    robot_udp_port: int = 5556
    robot_command_format: str = "LR {left:.1f} {right:.1f}"
    max_cross_track_error_cm: float = 8.0
    base_speed_pct: float = 38.0
    max_speed_pct: float = 80.0
    heading_kp: float = 38.0
    xte_kp: float = 2.2
    max_heading_for_forward_rad: float = math.radians(70.0)
    min_send_interval_s: float = 0.02
    command_deadband_pct: float = 1.0
    manual_move_units: int = 5
    manual_move_speed: int = 40
    manual_turn_degrees: int = 15
    manual_turn_speed: int = 30
    key_left_arrow: set[int] = dataclass_field(default_factory=lambda: {2424832, 65361, 63234, 81})
    key_up_arrow: set[int] = dataclass_field(default_factory=lambda: {2490368, 65362, 63232, 82})
    key_right_arrow: set[int] = dataclass_field(default_factory=lambda: {2555904, 65363, 63235, 83})
    key_down_arrow: set[int] = dataclass_field(default_factory=lambda: {2621440, 65364, 63233, 84})


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
