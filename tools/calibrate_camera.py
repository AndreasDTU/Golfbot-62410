#!/usr/bin/env python3
"""Interactive camera calibration tool using a checkerboard."""

from __future__ import annotations

import sys
from dataclasses import dataclass

import cv2
import numpy as np


CHECKERBOARD_INNER_CORNERS = (6, 8)
SQUARE_SIZE_MM = 25.0
CAMERA_INDEX = 0
OUTPUT_FILE = "calibration_data.npz"
MIN_CALIBRATION_IMAGES = 3
MIN_BOARD_AREA_RATIO = 0.08
MIN_CENTER_OFFSET_RATIO = 0.12
MIN_CAPTURE_DELTA_RATIO = 0.08
MIN_ANGLE_DELTA_DEG = 7.0
FISHEYE_FLAGS = (
    cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
    | cv2.fisheye.CALIB_CHECK_COND
    | cv2.fisheye.CALIB_FIX_SKEW
)
FISHEYE_FLAGS_FALLBACK = (
    cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
    | cv2.fisheye.CALIB_FIX_SKEW
)

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


@dataclass(frozen=True)
class CaptureQuality:
    accepted: bool
    summary: str
    detail: str
    color: tuple[int, int, int]
    center_ratio: float
    area_ratio: float
    angle_deg: float


def normalize_corners(corners: np.ndarray) -> np.ndarray:
    """Return corners in OpenCV chessboard shape (N, 1, 2)."""
    return np.ascontiguousarray(corners.reshape(-1, 1, 2), dtype=np.float64)


def corners_for_draw(corners: np.ndarray) -> np.ndarray:
    """Return corners formatted for cv2.drawChessboardCorners."""
    return np.ascontiguousarray(corners.reshape(-1, 1, 2), dtype=np.float32)


def build_object_points() -> np.ndarray:
    """Create checkerboard corner coordinates in millimeters for fisheye calibration."""
    cols, rows = CHECKERBOARD_INNER_CORNERS
    object_points = np.zeros((rows * cols, 1, 3), np.float64)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    object_points[:, 0, :2] = grid * float(SQUARE_SIZE_MM)
    return object_points


def detect_checkerboard(gray: np.ndarray) -> tuple[bool, np.ndarray | None]:
    found, corners = cv2.findChessboardCorners(
        gray,
        CHECKERBOARD_INNER_CORNERS,
        CHESSBOARD_FLAGS,
    )
    if not found or corners is None:
        found_sb, corners_sb = cv2.findChessboardCornersSB(gray, CHECKERBOARD_INNER_CORNERS, None)
        if not found_sb or corners_sb is None:
            return False, None
        corners = corners_sb

    refined_corners = cv2.cornerSubPix(
        gray,
        corners_for_draw(corners),
        winSize=(11, 11),
        zeroZone=(-1, -1),
        criteria=SUBPIX_CRITERIA,
    )
    return True, normalize_corners(refined_corners)


def board_metrics(corners: np.ndarray, image_size: tuple[int, int]) -> tuple[float, float, float]:
    cols, rows = CHECKERBOARD_INNER_CORNERS
    points = corners.reshape(rows, cols, 2)
    tl = points[0, 0]
    tr = points[0, cols - 1]
    br = points[rows - 1, cols - 1]
    bl = points[rows - 1, 0]
    quad = np.array([tl, tr, br, bl], dtype=np.float64)

    width, height = image_size
    frame_center = np.array([width / 2.0, height / 2.0], dtype=np.float64)
    board_center = quad.mean(axis=0)
    diagonal = float(np.hypot(width, height))
    center_ratio = float(np.linalg.norm(board_center - frame_center) / diagonal)
    area_ratio = float(cv2.contourArea(quad.astype(np.float32)) / (width * height))
    top_edge = tr - tl
    angle_deg = float(np.degrees(np.arctan2(top_edge[1], top_edge[0])))
    return center_ratio, area_ratio, angle_deg


def assess_capture_quality(
    corners: np.ndarray,
    image_size: tuple[int, int],
    accepted_captures: list[CaptureQuality],
) -> CaptureQuality:
    center_ratio, area_ratio, angle_deg = board_metrics(corners, image_size)

    if area_ratio < MIN_BOARD_AREA_RATIO:
        return CaptureQuality(
            accepted=False,
            summary="For lille checkerboard",
            detail="Gaa taettere paa eller brug et stoerre board.",
            color=(0, 0, 255),
            center_ratio=center_ratio,
            area_ratio=area_ratio,
            angle_deg=angle_deg,
        )

    if center_ratio < MIN_CENTER_OFFSET_RATIO and accepted_captures:
        return CaptureQuality(
            accepted=False,
            summary="For centralt billede",
            detail="Flyt boardet mod kanter eller hjoerner for bedre daekning.",
            color=(0, 165, 255),
            center_ratio=center_ratio,
            area_ratio=area_ratio,
            angle_deg=angle_deg,
        )

    for previous in accepted_captures:
        center_delta = abs(center_ratio - previous.center_ratio)
        area_delta = abs(area_ratio - previous.area_ratio)
        angle_delta = abs(angle_deg - previous.angle_deg)
        if center_delta < MIN_CAPTURE_DELTA_RATIO and area_delta < 0.03 and angle_delta < MIN_ANGLE_DELTA_DEG:
            return CaptureQuality(
                accepted=False,
                summary="For lig et tidligere billede",
                detail="Aendr placering, afstand eller rotation tydeligere.",
                color=(0, 165, 255),
                center_ratio=center_ratio,
                area_ratio=area_ratio,
                angle_deg=angle_deg,
            )

    return CaptureQuality(
        accepted=True,
        summary="Billede egnet",
        detail="Board er stort nok og giver ny geometri.",
        color=(0, 255, 0),
        center_ratio=center_ratio,
        area_ratio=area_ratio,
        angle_deg=angle_deg,
    )


def trial_calibration(
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
    image_size: tuple[int, int] | None,
) -> tuple[bool, str]:
    if image_size is None or len(object_points) < MIN_CALIBRATION_IMAGES:
        remaining = max(MIN_CALIBRATION_IMAGES - len(object_points), 0)
        return False, f"Mangler mindst {remaining} gyldige billeder"

    K = np.zeros((3, 3), dtype=np.float64)
    D = np.zeros((4, 1), dtype=np.float64)
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in object_points]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in object_points]

    try:
        rms, _, _, _, _ = cv2.fisheye.calibrate(
            object_points,
            image_points,
            image_size,
            K,
            D,
            rvecs,
            tvecs,
            FISHEYE_FLAGS,
            SUBPIX_CRITERIA,
        )
        return True, f"Kalibrering ser stabil ud (RMS {rms:.3f})"
    except cv2.error:
        return False, "Datasaettet er endnu ikke stabilt nok til fisheye-kalibrering"


def draw_overlay(
    frame: np.ndarray,
    captured_count: int,
    live_status: str,
    live_color: tuple[int, int, int],
    dataset_status: str,
    dataset_color: tuple[int, int, int],
) -> np.ndarray:
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
    cv2.putText(
        overlay,
        f"Live check: {live_status}",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        live_color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        overlay,
        f"Kalibreringsstatus: {dataset_status}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        dataset_color,
        2,
        cv2.LINE_AA,
    )
    return overlay


def main() -> int:
    cv2.ocl.setUseOpenCL(False)

    object_point_template = build_object_points()
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    accepted_captures: list[CaptureQuality] = []

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Kunne ikke aabne kamera {CAMERA_INDEX}", file=sys.stderr)
        return 1

    image_size: tuple[int, int] | None = None
    should_calibrate = False
    last_live_status = "Vis checkerboardet i kameraet"
    last_live_color = (0, 255, 255)
    dataset_ready = False
    dataset_status = f"Mangler mindst {MIN_CALIBRATION_IMAGES} gyldige billeder"
    dataset_color = (0, 165, 255)

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("Kunne ikke laese frame fra kameraet", file=sys.stderr)
                return 1

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found_live, live_corners = detect_checkerboard(gray)
            if found_live and live_corners is not None:
                current_image_size = (gray.shape[1], gray.shape[0])
                live_quality = assess_capture_quality(live_corners, current_image_size, accepted_captures)
                last_live_status = f"{live_quality.summary}: {live_quality.detail}"
                last_live_color = live_quality.color
            else:
                live_quality = None
                last_live_status = "Checkerboard ikke fundet"
                last_live_color = (0, 0, 255)

            preview = draw_overlay(
                frame,
                len(image_points),
                last_live_status,
                last_live_color,
                dataset_status,
                dataset_color,
            )
            if found_live and live_corners is not None:
                cv2.drawChessboardCorners(
                    preview,
                    CHECKERBOARD_INNER_CORNERS,
                    corners_for_draw(live_corners),
                    True,
                )
            cv2.imshow(WINDOW_NAME, preview)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("c"):
                should_calibrate = True
                break

            if key != ord(" "):
                continue

            if not found_live or live_corners is None or live_quality is None:
                print("Checkerboard ikke fundet, prov igen")
                continue

            if not live_quality.accepted:
                print(f"Billede afvist: {live_quality.summary}. {live_quality.detail}")
                continue

            object_points.append(object_point_template.copy())
            image_points.append(live_corners.astype(np.float64))
            image_size = (gray.shape[1], gray.shape[0])
            accepted_captures.append(live_quality)
            dataset_ready, dataset_status = trial_calibration(object_points, image_points, image_size)
            dataset_color = (0, 255, 0) if dataset_ready else (0, 165, 255)
            print(f"Billede accepteret: {live_quality.summary}. {live_quality.detail}")
            print(f"Kalibreringsstatus: {dataset_status}")

            confirmation = draw_overlay(
                frame,
                len(image_points),
                f"Capture gemt: {live_quality.summary}",
                (0, 255, 0),
                dataset_status,
                dataset_color,
            )
            cv2.drawChessboardCorners(
                confirmation,
                CHECKERBOARD_INNER_CORNERS,
                corners_for_draw(live_corners),
                True,
            )
            cv2.imshow(WINDOW_NAME, confirmation)
            cv2.waitKey(500)

        if not should_calibrate:
            return 0

        if not object_points or image_size is None:
            print("Ingen gyldige checkerboard-billeder blev taget. Kalibrering afbrudt.")
            return 1

        if len(object_points) < MIN_CALIBRATION_IMAGES:
            print(
                f"For faa billeder til stabil fisheye-kalibrering: {len(object_points)}. "
                f"Tag mindst {MIN_CALIBRATION_IMAGES} billeder med forskellige vinkler og placeringer."
            )
            return 1

        if not dataset_ready:
            print(
                "Datasaettet bestod ikke den loebende kalibreringstest. "
                "Tag flere eller bedre checkerboard-billeder foer endelig kalibrering."
            )
            return 1

        def run_fisheye_calibration(flags: int):
            K = np.zeros((3, 3), dtype=np.float64)
            D = np.zeros((4, 1), dtype=np.float64)
            rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in object_points]
            tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in object_points]
            return cv2.fisheye.calibrate(
                object_points,
                image_points,
                image_size,
                K,
                D,
                rvecs,
                tvecs,
                flags,
                SUBPIX_CRITERIA,
            )

        try:
            rms, K, D, _, _ = run_fisheye_calibration(FISHEYE_FLAGS)
        except cv2.error as exc:
            if "CALIB_CHECK_COND" not in str(exc):
                raise

            print("Fisheye-kalibrering fejlede med CALIB_CHECK_COND.")
            print(
                "Det betyder normalt, at checkerboard-billederne er for ens eller for svage "
                "til en stabil løsning. Prøv flere billeder med tydelig tilt, rotation og "
                "placering tæt på hjørner og kanter."
            )
            print("Proever fallback uden CALIB_CHECK_COND for at se, om datasettet stadig kan bruges.")
            rms, K, D, _, _ = run_fisheye_calibration(FISHEYE_FLAGS_FALLBACK)

        print(f"Reprojection error (RMS): {rms}")
        print(f"Antal brugte checkerboard-billeder: {len(object_points)}")
        print("Fisheye Camera Matrix (K):")
        print(K)
        print("Fisheye Distortion Coefficients (D):")
        print(D)

        np.savez(
            OUTPUT_FILE,
            K=K,
            D=D,
            image_size=np.array(image_size, dtype=np.int32),
        )
        print(f"Kalibreringsdata gemt i {OUTPUT_FILE}")
        return 0
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
