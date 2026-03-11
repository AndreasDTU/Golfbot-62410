#!/usr/bin/env python3
"""Undistort Bane.png using saved camera calibration data."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from camera.imageprocessing import undistort_with_calibration


INPUT_IMAGE = REPO_ROOT / "Bane.png"
CALIBRATION_FILE = REPO_ROOT / "calibration_data.npz"
OUTPUT_IMAGE = REPO_ROOT / "Bane_undistorted.png"


def main() -> int:
    if not INPUT_IMAGE.exists():
        print(f"Inputbillede ikke fundet: {INPUT_IMAGE}", file=sys.stderr)
        return 1

    if not CALIBRATION_FILE.exists():
        print(f"Kalibreringsfil ikke fundet: {CALIBRATION_FILE}", file=sys.stderr)
        return 1

    image = cv2.imread(str(INPUT_IMAGE), cv2.IMREAD_COLOR)
    if image is None:
        print(f"Kunne ikke laese billede: {INPUT_IMAGE}", file=sys.stderr)
        return 1

    undistorted = undistort_with_calibration(image, str(CALIBRATION_FILE))

    if not cv2.imwrite(str(OUTPUT_IMAGE), undistorted):
        print(f"Kunne ikke gemme output: {OUTPUT_IMAGE}", file=sys.stderr)
        return 1

    print(f"Undistorted billede gemt i: {OUTPUT_IMAGE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
