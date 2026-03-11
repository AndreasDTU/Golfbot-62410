#!/usr/bin/env python3
"""Interactive camera calibration tool using a checkerboard."""

from __future__ import annotations

import sys

import cv2
import numpy as np


CHECKERBOARD_INNER_CORNERS = (6, 8)
SQUARE_SIZE_MM = 25.0
CAMERA_INDEX = 0
OUTPUT_FILE = "calibration_data.npz"

WINDOW_NAME = "Camera Calibration"
INSTRUCTION_TEXT = (
    "Tryk MELLEMRUM for at tage billede. Tryk 'C' for at kalibrere. Tryk 'Q' for at afslutte."
)
SUBPIX_CRITERIA = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001,
)
CHESSBOARD_FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE


def build_object_points() -> np.ndarray:
    """Create checkerboard corner coordinates in millimeters."""
    cols, rows = CHECKERBOARD_INNER_CORNERS
    object_points = np.zeros((rows * cols, 3), np.float32)
    object_points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * SQUARE_SIZE_MM
    return object_points


def draw_overlay(frame: np.ndarray, captured_count: int) -> np.ndarray:
    """Add usage instructions and capture count to the live preview."""
    overlay = frame.copy()
    cv2.putText(
        overlay,
        INSTRUCTION_TEXT,
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        overlay,
        f"Billeder taget: {captured_count}",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return overlay


def main() -> int:
    object_point_template = build_object_points()
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Kunne ikke aabne kamera {CAMERA_INDEX}", file=sys.stderr)
        return 1

    image_size: tuple[int, int] | None = None
    should_calibrate = False

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("Kunne ikke laese frame fra kameraet", file=sys.stderr)
                return 1

            preview = draw_overlay(frame, len(image_points))
            cv2.imshow(WINDOW_NAME, preview)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("c"):
                should_calibrate = True
                break

            if key != ord(" "):
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(
                gray,
                CHECKERBOARD_INNER_CORNERS,
                CHESSBOARD_FLAGS,
            )

            if not found or corners is None:
                print("Checkerboard ikke fundet, prov igen")
                continue

            refined_corners = cv2.cornerSubPix(
                gray,
                corners,
                winSize=(11, 11),
                zeroZone=(-1, -1),
                criteria=SUBPIX_CRITERIA,
            )

            object_points.append(object_point_template.copy())
            image_points.append(refined_corners)
            image_size = (gray.shape[1], gray.shape[0])

            confirmation = draw_overlay(frame, len(image_points))
            cv2.drawChessboardCorners(
                confirmation,
                CHECKERBOARD_INNER_CORNERS,
                refined_corners,
                found,
            )
            cv2.imshow(WINDOW_NAME, confirmation)
            cv2.waitKey(500)

        if not should_calibrate:
            return 0

        if not object_points or image_size is None:
            print("Ingen gyldige checkerboard-billeder blev taget. Kalibrering afbrudt.")
            return 1

        rms, camera_matrix, distortion_coeffs, _, _ = cv2.calibrateCamera(
            object_points,
            image_points,
            image_size,
            None,
            None,
        )

        print(f"Reprojection error (RMS): {rms}")
        print("Camera Matrix:")
        print(camera_matrix)
        print("Distortion Coefficients:")
        print(distortion_coeffs)

        np.savez(
            OUTPUT_FILE,
            camera_matrix=camera_matrix,
            distortion_coeffs=distortion_coeffs,
        )
        print(f"Kalibreringsdata gemt i {OUTPUT_FILE}")
        return 0
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
