import argparse
import cv2
import sys
import platform
from pathlib import Path
import numpy as np

from config import AppConfig
from gui.MainGUI import MainGui

from perception.vision.pipeline import VisionPipeline
from localization.localization import RobotPoseEstimator
from config import AppConfig
from perception.vision.debug import DebugRenderer
from perception.vision.pipeline import VisionPipeline
from perception.vision.detection import BallDetector

class _NullBallDetector(BallDetector):
    """Fallback detector when YOLO model is unavailable."""

    def detect(self, frame_bgr, params, camera_center_pixels):
        return [], [], {"white": np.zeros(frame_bgr.shape[:2], dtype=np.uint8),
                        "orange": np.zeros(frame_bgr.shape[:2], dtype=np.uint8)}

REPO_ROOT = Path(__file__).resolve().parents[0]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GolfBot Main GUI — camera + schematic view.")
    parser.add_argument("--camera", type=int, default=None,
                        help="Camera device index (default from CameraConfig).")
    parser.add_argument("--no-camera", action="store_true",
                        help="Run without camera (schematic-only testing).")
    parser.add_argument("--image", type=str, default=None,
                        help="Use a static image instead of live camera.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = AppConfig.from_repo_root(REPO_ROOT)

    try:
        pipeline = VisionPipeline(app_config=config, normalize_illumination=True)
    except Exception as exc:
        print(f"Warning: Could not build full pipeline ({exc}). Trying without YOLO.", file=sys.stderr)
        pipeline = VisionPipeline(
            app_config=config,
            ball_detector=_NullBallDetector(),
            normalize_illumination=True,
        )

    mapper = pipeline.mapper
    pose_estimator = RobotPoseEstimator(config.field, config.robot, mapper, planner_config=config.planner)
    renderer = DebugRenderer(config.field, config.windows, config.robot, config.drive, mapper, config.planner)

    camera = None
    static_image = None

    if args.image is not None:
        img = cv2.imread(args.image)
        if img is None:
            print(f"Could not read image: {args.image}", file=sys.stderr)
            return 1
        static_image = img
    elif not args.no_camera:
        cam_index = args.camera if args.camera is not None else config.camera.camera_index
        if platform.system() == "Windows":
            camera = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        else:
            camera = cv2.VideoCapture(cam_index)

        if not camera.isOpened():
            print(f"Could not open camera {cam_index}", file=sys.stderr)
            return 1

    gui = MainGui(
        config=config,
        pipeline=pipeline,
        pose_estimator=pose_estimator,
        renderer=renderer,
        camera=camera,
        static_image=static_image,
    )
    gui.run()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
