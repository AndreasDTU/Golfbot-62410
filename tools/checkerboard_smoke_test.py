#!/usr/bin/env python3
"""Single-image checkerboard smoke test.

This tool is intentionally fixed to one input image and fixed output paths.
It validates checkerboard detection and writes deterministic debug artifacts.
"""

from __future__ import annotations

import json
import sys
import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

INPUT_IMAGE = Path("/Users/alex/PycharmProjects/Golfbot-62410/test/images/checkerboard_test.png")
OVERLAY_OUTPUT = Path("/Users/alex/PycharmProjects/Golfbot-62410/test/artifacts/checkerboard_overlay.png")
REPORT_OUTPUT = Path("/Users/alex/PycharmProjects/Golfbot-62410/test/artifacts/checkerboard_report.json")
SIDE_BY_SIDE_OUTPUT = Path(
    "/Users/alex/PycharmProjects/Golfbot-62410/test/artifacts/checkerboard_side_by_side.png"
)
# Fixed OpenCV inner-corner pattern for this test image.
# The visual board is 30x16 tiles with half-tile rows top/bottom,
# which maps to a detectable inner-corner grid of 29x17.
PATTERN = (29, 17)


@dataclass(frozen=True)
class CheckerboardReport:
    input_path: str
    pattern_inner_corners: list[int]
    image_width: int
    image_height: int
    found: bool
    detected_corner_count: int
    status: str
    timestamp_utc: str
    provisional_rms_error: float | None
    side_by_side_path: str | None


def _write_report(report: CheckerboardReport) -> None:
    REPORT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUTPUT.write_text(json.dumps(asdict(report), indent=2) + "\n", encoding="utf-8")


def _add_title(image_bgr, text, cv2):
    labeled = image_bgr.copy()
    cv2.putText(
        labeled,
        text,
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return labeled


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-image checkerboard smoke test with side-by-side undistortion preview."
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Do not open the preview window (still writes output files).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        import cv2
    except ModuleNotFoundError:
        sys.stderr.write(
            "Missing dependency: cv2. Install into this interpreter "
            f"({sys.executable}) with: {sys.executable} -m pip install opencv-python numpy\n"
        )
        return 2

    from camera.calibration import (
        calibrate_from_single_checkerboard,
        draw_checkerboard_overlay,
        find_checkerboard_corners,
        make_side_by_side,
        undistort_image,
    )

    image_bgr = cv2.imread(str(INPUT_IMAGE), cv2.IMREAD_COLOR)
    if image_bgr is None:
        sys.stderr.write(f"Failed to read fixed input image: {INPUT_IMAGE}\n")
        return 2

    found, corners = find_checkerboard_corners(image_bgr=image_bgr, pattern=PATTERN)
    overlay = draw_checkerboard_overlay(
        image_bgr=image_bgr,
        pattern=PATTERN,
        corners=corners,
        found=found,
    )

    OVERLAY_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OVERLAY_OUTPUT), overlay)

    detected_corner_count = 0 if corners is None else int(corners.shape[0])
    provisional_rms_error = None
    side_by_side_path = None

    if found and corners is not None:
        provisional_rms_error, camera_matrix, dist_coeffs = calibrate_from_single_checkerboard(
            image_bgr=image_bgr,
            corners=corners,
            pattern=PATTERN,
            square_size=1.0,
        )
        undistorted = undistort_image(image_bgr, camera_matrix, dist_coeffs)
        left = _add_title(image_bgr, "Original", cv2)
        right = _add_title(undistorted, "Provisional Undistorted", cv2)
        side_by_side = make_side_by_side(left, right)
        cv2.imwrite(str(SIDE_BY_SIDE_OUTPUT), side_by_side)
        side_by_side_path = str(SIDE_BY_SIDE_OUTPUT)

        if not args.no_gui:
            try:
                cv2.imshow("Checkerboard Undistortion Preview", side_by_side)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            except cv2.error:
                # Keep CLI deterministic and usable in headless environments.
                sys.stderr.write(
                    "GUI preview could not be opened in this environment. "
                    "Side-by-side image was saved to disk.\n"
                )

    report = CheckerboardReport(
        input_path=str(INPUT_IMAGE),
        pattern_inner_corners=[PATTERN[0], PATTERN[1]],
        image_width=int(image_bgr.shape[1]),
        image_height=int(image_bgr.shape[0]),
        found=bool(found),
        detected_corner_count=detected_corner_count,
        status="PASS" if found else "FAIL",
        timestamp_utc=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        provisional_rms_error=provisional_rms_error,
        side_by_side_path=side_by_side_path,
    )
    _write_report(report)

    if found:
        print(f"PASS: checkerboard found. corners={detected_corner_count}")
        print(f"Overlay: {OVERLAY_OUTPUT}")
        if side_by_side_path is not None:
            print(f"Preview: {SIDE_BY_SIDE_OUTPUT}")
        print(f"Report:  {REPORT_OUTPUT}")
        return 0

    print("FAIL: checkerboard not found.")
    print(f"Overlay: {OVERLAY_OUTPUT}")
    print(f"Report:  {REPORT_OUTPUT}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
