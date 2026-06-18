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


def _auto_hsv(frame: np.ndarray, cx: int, cy: int, crop_size: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Compute per-crop HSV bounds via Otsu on whiteness score (V − S)."""
    half = crop_size // 2
    h, w = frame.shape[:2]
    x1, y1 = max(0, cx - half), max(0, cy - half)
    x2, y2 = min(w, cx + half), min(h, cy + half)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)
    h_ch, s_ch, v_ch = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    whiteness = np.clip(v_ch - s_ch, 0, 255).astype(np.uint8)
    _, ball_mask = cv2.threshold(whiteness, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ball_pixels = ball_mask > 0
    if ball_pixels.sum() < 4:
        return None
    h_b, s_b, v_b = h_ch[ball_pixels], s_ch[ball_pixels], v_ch[ball_pixels]
    lower = np.array([max(0,   int(h_b.mean() - 1.5 * h_b.std())),
                      max(0,   int(s_b.mean() - 1.5 * s_b.std())),
                      max(0,   int(v_b.mean() - 1.5 * v_b.std()))], dtype=np.uint8)
    upper = np.array([min(180, int(h_b.mean() + 1.5 * h_b.std())),
                      min(255, int(s_b.mean() + 1.5 * s_b.std())),
                      min(255, int(v_b.mean() + 1.5 * v_b.std()))], dtype=np.uint8)
    return lower, upper


def _check_crop(frame: np.ndarray, cx: int, cy: int, params: dict) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray]:
    """Return (present, crop_bgr, manual_mask_bgr, auto_mask_bgr)."""
    half = params["crop_size"] // 2
    h, w = frame.shape[:2]
    x1, y1 = max(0, cx - half), max(0, cy - half)
    x2, y2 = min(w, cx + half), min(h, cy + half)
    cs = params["crop_size"]
    blank = np.zeros((cs, cs, 3), dtype=np.uint8)
    if x2 <= x1 or y2 <= y1:
        return False, blank, blank, blank

    crop = frame[y1:y2, x1:x2].copy()
    hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lower = np.array([params["h_min"], params["s_min"], params["v_min"]], dtype=np.uint8)
    upper = np.array([params["h_max"], params["s_max"], params["v_max"]], dtype=np.uint8)
    mask  = cv2.inRange(hsv, lower, upper)

    # Auto HSV mask
    auto = _auto_hsv(frame, cx, cy, params["crop_size"])
    if auto is not None:
        auto_mask = cv2.inRange(hsv, auto[0], auto[1])
    else:
        auto_mask = np.zeros_like(mask)

    # Pad to square if clipped at frame edge
    if crop.shape[0] != cs or crop.shape[1] != cs:
        def _pad(arr: np.ndarray, fill: int = 0) -> np.ndarray:
            p = np.full((cs, cs) if arr.ndim == 2 else (cs, cs, arr.shape[2]), fill, dtype=arr.dtype)
            p[:arr.shape[0], :arr.shape[1]] = arr
            return p
        crop = _pad(crop)
        mask = _pad(mask)
        auto_mask = _pad(auto_mask)

    fraction = float(np.count_nonzero(mask)) / (cs * cs)
    present = fraction >= 0.03
    return present, crop, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), cv2.cvtColor(auto_mask, cv2.COLOR_GRAY2BGR)


def _build_right_panel(frame: np.ndarray, balls: list[tuple[int, int]], params: dict) -> np.ndarray:
    """Build a vertical strip: crop | manual mask | auto mask per ball."""
    cs = params["crop_size"]
    row_h = cs + 4
    panel_h = max(40, row_h * max(1, len(balls)) + 24)
    panel = np.full((panel_h, PANEL_W, 3), 30, dtype=np.uint8)

    cv2.putText(panel, "crop | manual | auto", (8, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)

    for i, (cx, cy) in enumerate(balls):
        present, crop, mask_bgr, auto_bgr = _check_crop(frame, cx, cy, params)
        y0 = 20 + i * row_h
        if y0 + cs > panel_h:
            break
        col = 2
        panel[y0:y0 + cs, col:col + cs] = crop;        col += cs + 3
        panel[y0:y0 + cs, col:col + cs] = mask_bgr;    col += cs + 3
        panel[y0:y0 + cs, col:col + cs] = auto_bgr

        auto = _auto_hsv(frame, cx, cy, params["crop_size"])
        color = (60, 200, 60) if present else (60, 60, 200)
        label = f"#{i+1} {'OK' if present else 'MISS'}"
        if auto is not None:
            auto_label = f"A H{auto[0][0]}-{auto[1][0]} S{auto[0][1]}-{auto[1][1]} V{auto[0][2]}-{auto[1][2]}"
        else:
            auto_label = "auto: n/a"
        right_x = cs * 3 + 12
        cv2.putText(panel, label,      (right_x, y0 + cs // 2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)
        cv2.putText(panel, auto_label, (right_x, y0 + cs // 2 + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (140, 180, 140), 1)

    return panel


def _draw_overlay(canvas: np.ndarray, balls: list[tuple[int, int]], params: dict) -> None:
    half = params["crop_size"] // 2
    for cx, cy in balls:
        present, _, _, _ = _check_crop(canvas, cx, cy, params)
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
