"""Calibration helpers for checkerboard smoke testing.

This module intentionally stays lightweight and deterministic.
It supports checkerboard detection, overlay rendering, and a provisional
single-image undistortion path for visualization only.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

try:
    import cv2
except ModuleNotFoundError:  # pragma: no cover - handled by caller scripts/tests
    cv2 = None  # type: ignore[assignment]


PatternType = Tuple[int, int]

DEFAULT_PATTERN: PatternType = (11, 8)
CHESSBOARD_FLAGS = (
    cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE if cv2 is not None else 0
)
SUBPIX_CRITERIA = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001,
) if cv2 is not None else (0, 0, 0.0)


def _require_cv2() -> None:
    if cv2 is None:
        raise ModuleNotFoundError(
            "OpenCV (cv2) is required for checkerboard testing. "
            "Install with: pip install opencv-python"
        )


def find_checkerboard_corners(
    image_bgr: Any, pattern: PatternType = DEFAULT_PATTERN
) -> Tuple[bool, Optional[Any]]:
    """Detect checkerboard corners using fixed deterministic parameters.

    Uses classic corner detection first, then deterministic SB fallback for
    harder synthetic or strongly distorted boards.
    """
    _require_cv2()

    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("image_bgr must be a non-empty image array")

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, pattern, CHESSBOARD_FLAGS)

    if not found or corners is None:
        # Deterministic fallback for challenging patterns.
        found_sb, corners_sb = cv2.findChessboardCornersSB(gray, pattern, None)
        if found_sb and corners_sb is not None:
            return True, corners_sb
        return False, None

    refined = cv2.cornerSubPix(
        gray,
        corners,
        winSize=(11, 11),
        zeroZone=(-1, -1),
        criteria=SUBPIX_CRITERIA,
    )
    return True, refined


def draw_checkerboard_overlay(
    image_bgr: Any,
    pattern: PatternType,
    corners: Optional[Any],
    found: bool,
) -> Any:
    """Draw checkerboard corners on a copy of the image and return it."""
    _require_cv2()

    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("image_bgr must be a non-empty image array")

    overlay = image_bgr.copy()
    if corners is not None:
        cv2.drawChessboardCorners(overlay, pattern, corners, found)
    return overlay


def calibrate_from_single_checkerboard(
    image_bgr: Any,
    corners: Any,
    pattern: PatternType = DEFAULT_PATTERN,
    square_size: float = 1.0,
) -> Tuple[float, Any, Any]:
    """Compute provisional calibration from one checkerboard image.

    This is only suitable for smoke tests and preview visualization.
    """
    _require_cv2()

    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("image_bgr must be a non-empty image array")
    if corners is None:
        raise ValueError("corners must not be None when calibrating")
    if square_size <= 0:
        raise ValueError("square_size must be > 0")

    import numpy as np

    cols, rows = pattern
    objp = np.zeros((rows * cols, 3), np.float32)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp[:, :2] = grid * float(square_size)

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    image_size = (gray.shape[1], gray.shape[0])
    rms, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
        [objp], [corners], image_size, None, None
    )
    return float(rms), camera_matrix, dist_coeffs


def undistort_image(image_bgr: Any, camera_matrix: Any, dist_coeffs: Any) -> Any:
    """Undistort image using supplied camera intrinsics."""
    _require_cv2()

    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("image_bgr must be a non-empty image array")
    return cv2.undistort(image_bgr, camera_matrix, dist_coeffs)


def make_side_by_side(left_bgr: Any, right_bgr: Any) -> Any:
    """Return a horizontal side-by-side image."""
    _require_cv2()

    if left_bgr is None or right_bgr is None:
        raise ValueError("left_bgr and right_bgr must not be None")
    if left_bgr.shape != right_bgr.shape:
        raise ValueError("left_bgr and right_bgr must have identical shape")

    import numpy as np

    return np.hstack((left_bgr, right_bgr))


def compute_inner_corner_quad(corners: Any, pattern: PatternType) -> Any:
    """Return TL, TR, BR, BL points from checkerboard inner corners."""
    _require_cv2()

    if corners is None:
        raise ValueError("corners must not be None")

    import numpy as np

    cols, rows = pattern
    expected = rows * cols
    if int(corners.shape[0]) != expected:
        raise ValueError(
            f"corner count mismatch: got {int(corners.shape[0])}, expected {expected} for pattern={pattern}"
        )

    grid = corners.reshape(rows, cols, 2).astype(np.float32)
    tl = grid[0, 0]
    tr = grid[0, cols - 1]
    br = grid[rows - 1, cols - 1]
    bl = grid[rows - 1, 0]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def warp_from_checkerboard_inner_corners(
    image_bgr: Any,
    corners: Any,
    pattern: PatternType,
    square_px: int = 40,
) -> Tuple[Any, Any, Any]:
    """Warp image to a fronto-parallel view based on checkerboard inner corners.

    Returns: (warped_image, homography_matrix, source_quad_points)
    """
    _require_cv2()

    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("image_bgr must be a non-empty image array")
    if square_px <= 0:
        raise ValueError("square_px must be > 0")

    import numpy as np

    cols, rows = pattern
    width = int((cols - 1) * square_px)
    height = int((rows - 1) * square_px)
    if width < 2 or height < 2:
        raise ValueError(
            f"invalid output size ({width}x{height}) for pattern={pattern}. "
            "Pattern must be at least 2x2 inner corners."
        )

    src_quad = compute_inner_corner_quad(corners=corners, pattern=pattern)
    dst_quad = np.array(
        [
            [0.0, 0.0],
            [float(width - 1), 0.0],
            [float(width - 1), float(height - 1)],
            [0.0, float(height - 1)],
        ],
        dtype=np.float32,
    )

    homography = cv2.getPerspectiveTransform(src_quad, dst_quad)
    warped = cv2.warpPerspective(image_bgr, homography, (width, height))
    return warped, homography, src_quad


def draw_quad_overlay(image_bgr: Any, quad_points: Any, color: Tuple[int, int, int] = (0, 255, 0)) -> Any:
    """Draw a quadrilateral overlay on top of the image."""
    _require_cv2()

    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("image_bgr must be a non-empty image array")
    if quad_points is None:
        raise ValueError("quad_points must not be None")

    import numpy as np

    overlay = image_bgr.copy()
    pts = quad_points.reshape(-1, 1, 2).astype(np.int32)
    cv2.polylines(overlay, [pts], True, color, 2, cv2.LINE_AA)
    return overlay
