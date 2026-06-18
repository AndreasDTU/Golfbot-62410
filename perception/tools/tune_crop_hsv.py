"""Standalone HSV crop tuning tool.

Load a saved topdown image (or grab one frame from the live camera),
click to place ball positions, then tune the HSV sliders until the
crop boxes turn green.  Press 'r' to reset ball positions, 'q' to quit.

Usage:
    python perception/tools/tune_crop_hsv.py                     # live camera, 1 frame
    python perception/tools/tune_crop_hsv.py path/to/topdown.png # static image
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Defaults (match gui/MainGUI.py _seed_geometry_params)
# ---------------------------------------------------------------------------
DEFAULTS = {
    "crop_size":      60,
    "h_min":           0,
    "h_max":         180,
    "s_min":           0,
    "s_max":          40,
    "v_min":         200,
    "v_max":         255,
}

WINDOW = "Crop HSV Tuner  [click=place ball | r=reset | q=quit]"
PANEL_W = 320   # right panel width for crop previews


def _grab_camera_frame() -> np.ndarray | None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return None
    for _ in range(5):          # let the camera warm up
        cap.read()
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def _check_crop(frame: np.ndarray, cx: int, cy: int, params: dict) -> tuple[bool, np.ndarray, np.ndarray]:
    """Return (present, crop_bgr, mask_bgr)."""
    half = params["crop_size"] // 2
    h, w = frame.shape[:2]
    x1, y1 = max(0, cx - half), max(0, cy - half)
    x2, y2 = min(w, cx + half), min(h, cy + half)
    if x2 <= x1 or y2 <= y1:
        blank = np.zeros((params["crop_size"], params["crop_size"], 3), dtype=np.uint8)
        return False, blank, blank

    crop = frame[y1:y2, x1:x2].copy()
    hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lower = np.array([params["h_min"], params["s_min"], params["v_min"]], dtype=np.uint8)
    upper = np.array([params["h_max"], params["s_max"], params["v_max"]], dtype=np.uint8)
    mask  = cv2.inRange(hsv, lower, upper)
    fraction = float(np.count_nonzero(mask)) / mask.size

    # Pad crop to square if clipped at frame edge
    cs = params["crop_size"]
    if crop.shape[0] != cs or crop.shape[1] != cs:
        padded = np.zeros((cs, cs, 3), dtype=np.uint8)
        padded[:crop.shape[0], :crop.shape[1]] = crop
        crop = padded
        mask_pad = np.zeros((cs, cs), dtype=np.uint8)
        mask_pad[:mask.shape[0], :mask.shape[1]] = mask
        mask = mask_pad

    present = fraction >= 0.03          # same default as crop_min_pixel_fraction
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return present, crop, mask_bgr


def _build_right_panel(frame: np.ndarray, balls: list[tuple[int, int]], params: dict) -> np.ndarray:
    """Build a vertical strip showing each ball's crop and HSV mask."""
    cs = params["crop_size"]
    row_h = cs + 4
    panel_h = max(40, row_h * max(1, len(balls)) + 10)
    panel = np.full((panel_h, PANEL_W, 3), 30, dtype=np.uint8)

    cv2.putText(panel, "crop | mask", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

    for i, (cx, cy) in enumerate(balls):
        present, crop, mask_bgr = _check_crop(frame, cx, cy, params)
        y0 = 24 + i * row_h
        if y0 + cs > panel_h:
            break
        panel[y0:y0 + cs, 2:2 + cs] = crop
        panel[y0:y0 + cs, cs + 6:cs + 6 + cs] = mask_bgr
        color = (60, 200, 60) if present else (60, 60, 200)
        label = f"#{i+1} {'OK' if present else 'MISS'}"
        cv2.putText(panel, label, (cs * 2 + 12, y0 + cs // 2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    return panel


def _draw_overlay(canvas: np.ndarray, balls: list[tuple[int, int]], params: dict) -> None:
    half = params["crop_size"] // 2
    for cx, cy in balls:
        present, _, _ = _check_crop(canvas, cx, cy, params)
        color = (60, 200, 60) if present else (60, 60, 200)
        cv2.rectangle(canvas, (cx - half, cy - half), (cx + half, cy + half), color, 2, cv2.LINE_AA)
        cv2.drawMarker(canvas, (cx, cy), color, cv2.MARKER_CROSS, 10, 1, cv2.LINE_AA)


def main() -> None:
    # ---- Load image -------------------------------------------------------
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
        frame = cv2.imread(str(path))
        if frame is None:
            print(f"Could not load image: {path}")
            sys.exit(1)
        print(f"Loaded: {path}  ({frame.shape[1]}×{frame.shape[0]})")
    else:
        print("No image path given — grabbing one frame from camera 0…")
        frame = _grab_camera_frame()
        if frame is None:
            print("Could not open camera.")
            sys.exit(1)
        print(f"Captured frame: {frame.shape[1]}×{frame.shape[0]}")

    params = dict(DEFAULTS)
    balls: list[tuple[int, int]] = []

    # ---- Window & sliders -------------------------------------------------
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, frame.shape[1] + PANEL_W, max(400, frame.shape[0]))

    def nothing(_: int) -> None:
        pass

    cv2.createTrackbar("crop_size", WINDOW, params["crop_size"], 200, nothing)
    cv2.createTrackbar("H min",     WINDOW, params["h_min"],      180, nothing)
    cv2.createTrackbar("H max",     WINDOW, params["h_max"],      180, nothing)
    cv2.createTrackbar("S min",     WINDOW, params["s_min"],      255, nothing)
    cv2.createTrackbar("S max",     WINDOW, params["s_max"],      255, nothing)
    cv2.createTrackbar("V min",     WINDOW, params["v_min"],      255, nothing)
    cv2.createTrackbar("V max",     WINDOW, params["v_max"],      255, nothing)

    def on_click(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and x < frame.shape[1]:
            balls.append((x, y))

    cv2.setMouseCallback(WINDOW, on_click)

    print("Click on the topdown frame to place ball positions.")
    print("Adjust sliders until crop boxes turn green. Press 'q' to quit.")

    while True:
        # Read sliders
        cs = max(10, cv2.getTrackbarPos("crop_size", WINDOW))
        params.update({
            "crop_size": cs,
            "h_min": cv2.getTrackbarPos("H min", WINDOW),
            "h_max": cv2.getTrackbarPos("H max", WINDOW),
            "s_min": cv2.getTrackbarPos("S min", WINDOW),
            "s_max": cv2.getTrackbarPos("S max", WINDOW),
            "v_min": cv2.getTrackbarPos("V min", WINDOW),
            "v_max": cv2.getTrackbarPos("V max", WINDOW),
        })

        canvas = frame.copy()
        _draw_overlay(canvas, balls, params)

        right = _build_right_panel(frame, balls, params)
        if right.shape[0] != canvas.shape[0]:
            right = cv2.resize(right, (PANEL_W, canvas.shape[0]))
        combined = np.hstack([canvas, right])

        # Print current params in corner
        info = (f"H:{params['h_min']}-{params['h_max']}  "
                f"S:{params['s_min']}-{params['s_max']}  "
                f"V:{params['v_min']}-{params['v_max']}  "
                f"size:{params['crop_size']}")
        cv2.putText(combined, info, (8, combined.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

        cv2.imshow(WINDOW, combined)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        if key == ord('r'):
            balls.clear()
            print("Ball positions reset.")

    cv2.destroyAllWindows()

    # Print final values to copy into _seed_geometry_params
    print("\n--- Final HSV params (copy into _seed_geometry_params) ---")
    print(f'self.params.setdefault("crop_size",          {params["crop_size"]})')
    print(f'self.params.setdefault("crop_white_h_min",   {params["h_min"]})')
    print(f'self.params.setdefault("crop_white_h_max",   {params["h_max"]})')
    print(f'self.params.setdefault("crop_white_s_min",   {params["s_min"]})')
    print(f'self.params.setdefault("crop_white_s_max",   {params["s_max"]})')
    print(f'self.params.setdefault("crop_white_v_min",   {params["v_min"]})')
    print(f'self.params.setdefault("crop_white_v_max",   {params["v_max"]})')


if __name__ == "__main__":
    main()
