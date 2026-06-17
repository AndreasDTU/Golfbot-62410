#!/usr/bin/env python3
"""Live camera preview with fisheye undistortion applied."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_FILE = REPO_ROOT / "calibration_data.npz"
CAMERA_INDEX = 0
BALANCE = 0.0
WINDOW_NAME = "Live Undistort"


def build_maps(frame_size: tuple[int, int], calibration_file: Path, balance: float) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(str(calibration_file))
    K = data["K"]
    D = data["D"]
    calibration_image_size = tuple(int(v) for v in data["image_size"])

    if frame_size != calibration_image_size:
        raise ValueError(
            f"Kameraets frame-stoerrelse {frame_size} matcher ikke kalibreringsstoerrelsen "
            f"{calibration_image_size}. Brug samme oploesning som under kalibrering."
        )

    identity = np.eye(3, dtype=np.float64)
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K,
        D,
        frame_size,
        identity,
        balance=balance,
    )
    return cv2.fisheye.initUndistortRectifyMap(
        K,
        D,
        identity,
        new_K,
        frame_size,
        cv2.CV_32FC1,
    )


def add_label(frame: np.ndarray, text: str) -> np.ndarray:
    labeled = frame.copy()
    cv2.putText(
        labeled,
        text,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return labeled


def main() -> int:
    cv2.ocl.setUseOpenCL(False)

    if not CALIBRATION_FILE.exists():
        print(f"Kalibreringsfil ikke fundet: {CALIBRATION_FILE}", file=sys.stderr)
        return 1

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Kunne ikke aabne kamera {CAMERA_INDEX}", file=sys.stderr)
        return 1

    try:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("Kunne ikke laese foerste frame fra kameraet", file=sys.stderr)
            return 1

        frame_size = (int(frame.shape[1]), int(frame.shape[0]))
        try:
            map1, map2 = build_maps(frame_size, CALIBRATION_FILE, BALANCE)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Kunne ikke laese frame fra kameraet", file=sys.stderr)
                return 1

            undistorted = cv2.remap(
                frame,
                map1,
                map2,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )

            original_view = add_label(frame, "Original")
            undistorted_view = add_label(undistorted, f"Undistorted balance={BALANCE}")
            side_by_side = np.hstack((original_view, undistorted_view))
            cv2.imshow(WINDOW_NAME, side_by_side)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        return 0
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
