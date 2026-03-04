#!/usr/bin/env python3
"""Checkerboard perspective warp smoke test.

Detect checkerboard corners, estimate perspective transforms, and create
inspection outputs for:
1) warp(original)
2) warp(undistorted) when provisional undistortion is requested.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_INPUT = Path("/Users/alex/PycharmProjects/Golfbot-62410/test/images/checkerboard_test.png")
DEFAULT_ARTIFACT_DIR = Path("/Users/alex/PycharmProjects/Golfbot-62410/test/artifacts")
DEFAULT_PATTERN_COLS = 29
DEFAULT_PATTERN_ROWS = 17
DEFAULT_SQUARE_PX = 40


@dataclass(frozen=True)
class PerspectiveWarpReport:
    input_path: str
    pattern_inner_corners: list[int]
    square_px: int
    image_width: int
    image_height: int
    found: bool
    detected_corner_count: int
    status: str
    timestamp_utc: str
    warped_path: str | None
    warped_original_path: str | None
    warped_undistorted_path: str | None
    overlay_path: str
    side_by_side_path: str | None
    undistorted_path: str | None
    full_frame_path: str | None
    final_comparison_path: str | None
    provisional_rms_error: float | None
    warp_source: str | None
    homography: list[list[float]] | None


def _add_title(image_bgr, text: str, cv2):
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


def _make_2x2_grid(top_left, top_right, bottom_left, bottom_right):
    import numpy as np

    top = np.hstack((top_left, top_right))
    bottom = np.hstack((bottom_left, bottom_right))
    return np.vstack((top, bottom))


def _write_report(report_path: Path, report: PerspectiveWarpReport) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(asdict(report), indent=2) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Perspective warp smoke test from checkerboard inner corners."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input image path (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--pattern-cols",
        type=int,
        default=DEFAULT_PATTERN_COLS,
        help="Checkerboard inner corner columns.",
    )
    parser.add_argument(
        "--pattern-rows",
        type=int,
        default=DEFAULT_PATTERN_ROWS,
        help="Checkerboard inner corner rows.",
    )
    parser.add_argument(
        "--square-px",
        type=int,
        default=DEFAULT_SQUARE_PX,
        help="Output pixels per checkerboard inner-corner spacing.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR,
        help=f"Output artifact directory (default: {DEFAULT_ARTIFACT_DIR})",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Do not open preview window (still writes output files).",
    )
    parser.add_argument(
        "--full-frame",
        action="store_true",
        help="Generate one frame: Original | Undistorted | Primary Perspective Warp.",
    )
    parser.add_argument(
        "--final-test",
        action="store_true",
        help=(
            "Generate final 2x2 frame: "
            "Original, Undistorted, Warp(Original), Warp(Undistorted)."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    pattern = (args.pattern_cols, args.pattern_rows)

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
        draw_quad_overlay,
        find_checkerboard_corners,
        make_side_by_side,
        undistort_image,
        warp_from_checkerboard_inner_corners,
    )

    input_path = args.input.resolve()
    artifact_dir = args.artifact_dir.resolve()
    overlay_path = artifact_dir / "perspective_overlay.png"
    warped_path = artifact_dir / "perspective_warp.png"
    warped_original_path = artifact_dir / "perspective_warp_original.png"
    warped_undistorted_path = artifact_dir / "perspective_warp_undistorted.png"
    side_path = artifact_dir / "perspective_side_by_side.png"
    undistorted_path = artifact_dir / "perspective_undistorted.png"
    full_frame_path = artifact_dir / "perspective_full_frame.png"
    final_comparison_path = artifact_dir / "perspective_final_comparison.png"
    report_path = artifact_dir / "perspective_report.json"

    image_bgr = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        sys.stderr.write(f"Failed to read input image: {input_path}\n")
        return 2

    found, corners = find_checkerboard_corners(image_bgr=image_bgr, pattern=pattern)
    base_overlay = draw_checkerboard_overlay(
        image_bgr=image_bgr,
        pattern=pattern,
        corners=corners,
        found=found,
    )

    detected_corner_count = 0 if corners is None else int(corners.shape[0])
    homography_data = None
    warped_output_path = None
    warped_original_output_path = None
    warped_undistorted_output_path = None
    side_output_path = None
    undistorted_output_path = None
    full_output_path = None
    final_comparison_output_path = None
    provisional_rms_error = None
    warp_source = None
    final_overlay = base_overlay

    if found and corners is not None:
        warped_original, homography_original, src_quad = warp_from_checkerboard_inner_corners(
            image_bgr=image_bgr,
            corners=corners,
            pattern=pattern,
            square_px=args.square_px,
        )
        final_overlay = draw_quad_overlay(base_overlay, src_quad)

        artifact_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(warped_original_path), warped_original)
        warped_original_output_path = str(warped_original_path)

        warped_primary = warped_original
        homography_primary = homography_original
        warp_source = "original"
        undistorted = None
        warped_undistorted = None

        if args.full_frame or args.final_test:
            provisional_rms_error, camera_matrix, dist_coeffs = calibrate_from_single_checkerboard(
                image_bgr=image_bgr,
                corners=corners,
                pattern=pattern,
                square_size=1.0,
            )
            undistorted = undistort_image(image_bgr, camera_matrix, dist_coeffs)
            cv2.imwrite(str(undistorted_path), undistorted)
            undistorted_output_path = str(undistorted_path)

            found_und, corners_und = find_checkerboard_corners(undistorted, pattern=pattern)
            if found_und and corners_und is not None:
                warped_undistorted, homography_undistorted, _ = warp_from_checkerboard_inner_corners(
                    image_bgr=undistorted,
                    corners=corners_und,
                    pattern=pattern,
                    square_px=args.square_px,
                )
                cv2.imwrite(str(warped_undistorted_path), warped_undistorted)
                warped_undistorted_output_path = str(warped_undistorted_path)
                warped_primary = warped_undistorted
                homography_primary = homography_undistorted
                warp_source = "undistorted"
            else:
                warp_source = "original_fallback"

        cv2.imwrite(str(warped_path), warped_primary)
        warped_output_path = str(warped_path)
        homography_data = homography_primary.tolist()

        preview_left = _add_title(final_overlay, "Original + Detected Quad", cv2)
        preview_right = cv2.resize(warped_primary, (preview_left.shape[1], preview_left.shape[0]))
        preview_right = _add_title(preview_right, "Perspective Warp", cv2)
        side_by_side = make_side_by_side(preview_left, preview_right)
        cv2.imwrite(str(side_path), side_by_side)
        side_output_path = str(side_path)

        if args.full_frame and undistorted is not None:
            import numpy as np

            panel_width = image_bgr.shape[1]
            panel_height = image_bgr.shape[0]
            panel_original = _add_title(image_bgr, "Original", cv2)
            panel_undistorted = _add_title(undistorted, "Provisional Undistorted", cv2)
            panel_warp = cv2.resize(warped_primary, (panel_width, panel_height))
            panel_warp = _add_title(panel_warp, "Perspective Warp", cv2)
            full_frame = np.hstack((panel_original, panel_undistorted, panel_warp))
            cv2.imwrite(str(full_frame_path), full_frame)
            full_output_path = str(full_frame_path)

        if args.final_test and undistorted is not None:
            panel_width = image_bgr.shape[1]
            panel_height = image_bgr.shape[0]
            panel_original = _add_title(image_bgr, "Original", cv2)
            panel_undistorted = _add_title(undistorted, "Undistorted", cv2)
            panel_warp_original = cv2.resize(warped_original, (panel_width, panel_height))
            panel_warp_original = _add_title(panel_warp_original, "Warp (Original)", cv2)
            if warped_undistorted is not None:
                panel_warp_undistorted = cv2.resize(
                    warped_undistorted, (panel_width, panel_height)
                )
                panel_warp_undistorted = _add_title(
                    panel_warp_undistorted, "Warp (Undistorted)", cv2
                )
            else:
                panel_warp_undistorted = panel_warp_original.copy()
                panel_warp_undistorted = _add_title(
                    panel_warp_undistorted, "Warp (Undistorted Fallback)", cv2
                )

            final_comparison = _make_2x2_grid(
                panel_original,
                panel_undistorted,
                panel_warp_original,
                panel_warp_undistorted,
            )
            cv2.imwrite(str(final_comparison_path), final_comparison)
            final_comparison_output_path = str(final_comparison_path)

        if not args.no_gui:
            try:
                if args.final_test and final_comparison_output_path is not None:
                    frame_to_show = cv2.imread(str(final_comparison_path), cv2.IMREAD_COLOR)
                    cv2.imshow("Final Transform Comparison", frame_to_show)
                elif args.full_frame and full_output_path is not None:
                    frame_to_show = cv2.imread(str(full_frame_path), cv2.IMREAD_COLOR)
                    cv2.imshow("Full Perspective + Undistortion Preview", frame_to_show)
                else:
                    cv2.imshow("Perspective Warp Preview", side_by_side)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            except cv2.error:
                sys.stderr.write(
                    "GUI preview could not be opened in this environment. "
                    "Artifacts were still saved to disk.\n"
                )

    artifact_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(overlay_path), final_overlay)

    report = PerspectiveWarpReport(
        input_path=str(input_path),
        pattern_inner_corners=[pattern[0], pattern[1]],
        square_px=int(args.square_px),
        image_width=int(image_bgr.shape[1]),
        image_height=int(image_bgr.shape[0]),
        found=bool(found),
        detected_corner_count=detected_corner_count,
        status="PASS" if found else "FAIL",
        timestamp_utc=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        warped_path=warped_output_path,
        warped_original_path=warped_original_output_path,
        warped_undistorted_path=warped_undistorted_output_path,
        overlay_path=str(overlay_path),
        side_by_side_path=side_output_path,
        undistorted_path=undistorted_output_path,
        full_frame_path=full_output_path,
        final_comparison_path=final_comparison_output_path,
        provisional_rms_error=provisional_rms_error,
        warp_source=warp_source,
        homography=homography_data,
    )
    _write_report(report_path, report)

    if found:
        print(f"PASS: checkerboard found. corners={detected_corner_count}")
        print(f"Overlay:   {overlay_path}")
        print(f"Warp:      {warped_path}")
        if warped_original_output_path is not None:
            print(f"Warp Orig: {warped_original_path}")
        if warped_undistorted_output_path is not None:
            print(f"Warp Und:  {warped_undistorted_path}")
        print(f"Preview:   {side_path}")
        if undistorted_output_path is not None:
            print(f"Undist:    {undistorted_path}")
        if full_output_path is not None:
            print(f"Full:      {full_frame_path}")
        if final_comparison_output_path is not None:
            print(f"Final:     {final_comparison_path}")
        print(f"Report:    {report_path}")
        return 0

    print("FAIL: checkerboard not found.")
    print(f"Overlay: {overlay_path}")
    print(f"Report:  {report_path}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
