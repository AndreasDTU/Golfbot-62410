import argparse
import cv2
import sys
import platform
from pathlib import Path
import numpy as np
import multiprocessing as mp
from config import AppConfig
from gui.MainGUI import MainGui

from control.robot.robot_service import RemoteRobotCommander, RobotControlService
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

def _build_pipeline(config: AppConfig) -> tuple[VisionPipeline, RobotPoseEstimator, DebugRenderer]:
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
    pose_estimator = RobotPoseEstimator(config.field, config.robot, mapper)
    renderer = DebugRenderer(config.field, config.windows, config.robot, config.drive, mapper)
    return pipeline, pose_estimator, renderer

def _gui_process_main(
    repo_root: str,
    arg_values: dict[str, object],
    request_queue,
    response_queue,
) -> None:
    repo_root_path = Path(repo_root)
    config = AppConfig.from_repo_root(repo_root_path)
    pipeline, pose_estimator, renderer = _build_pipeline(config)

    args = argparse.Namespace(**arg_values)
    camera = None
    static_image = None

    if args.image is not None:
        img = cv2.imread(args.image)
        if img is None:
            print(f"Could not read image: {args.image}", file=sys.stderr)
            return
        static_image = img
    elif not args.no_camera:
        cam_index = args.camera if args.camera is not None else config.camera.camera_index
        if platform.system() == "Windows":
            camera = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        else:
            camera = cv2.VideoCapture(cam_index)

        if not camera.isOpened():
            print(f"Could not open camera {cam_index}", file=sys.stderr)
            return

    def commander_factory(**kwargs):
        return RemoteRobotCommander(request_queue, response_queue, **kwargs)

    gui = MainGui(
        config=config,
        pipeline=pipeline,
        pose_estimator=pose_estimator,
        renderer=renderer,
        commander_factory=commander_factory,
        camera=camera,
        static_image=static_image,
    )
    gui.run()


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = AppConfig.from_repo_root(REPO_ROOT)
    service = RobotControlService(config.connection, config.drive)
    service.start()

    try:
        status = service.status_queue.get(timeout=15.0)
        if status[0] == "error":
            print(f"Warning: robot service failed to connect ({status[1]})", file=sys.stderr)
    except Exception:
        print("Warning: robot service did not report ready state", file=sys.stderr)

    ctx = mp.get_context("spawn")
    gui_proc = ctx.Process(
        target=_gui_process_main,
        args=(str(REPO_ROOT), vars(args), service.request_queue, service.response_queue),
        daemon=False,
    )
    gui_proc.start()

    try:
        gui_proc.join()
    finally:
        service.close()

    return gui_proc.exitcode or 0

if __name__ == "__main__":
    raise SystemExit(main())
