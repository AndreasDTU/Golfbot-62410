# Camera Workflow Overview

Status: ACTIVE KEPT PERCEPTION WORKFLOW

This document describes the camera and top-down tools that currently exist
after the layer reorganization.

## Current Locations

```text
perception/camera/
perception/tools/
perception/vision/
localization/tools/robot_origin_calibration.py
calibration_data.npz
robot_calibration.json
```

The old root paths `camera/`, `vision/`, and `tools/` are not current source
locations.

## Typical Camera Workflow

1. Calibrate the camera with a checkerboard.
2. Save calibration constants in `calibration_data.npz`.
3. Use those constants to undistort images or live feed.
4. Choose a top-down method: ArUco markers, HSV-based arena frame, or manual
   four-point selection.
5. Warp the image to a top-down view.
6. Use manual four-point selection as fallback if automatic corner finding is
   unstable.

## Calibration

Main script:

```text
perception/tools/calibrate_camera.py
```

It captures checkerboard observations, runs OpenCV fisheye calibration, and
writes:

```text
calibration_data.npz
```

with `K`, `D`, and `image_size`.

## Undistortion

Shared helper:

```text
perception/camera/imageprocessing.py
```

Important functions:

- `imageprocessing(img, colorspace)`
- `undistort_with_calibration(img, calibration_file, balance=0.0)`

Still-image helper:

```text
perception/tools/undistort_bane.py
```

Live helper:

```text
perception/tools/live_undistort.py
```

## Top-Down Tools

ArUco-based top-down helper:

```text
perception/tools/auto_topdown_aruco.py
```

HSV/debug top-down helper:

```text
perception/tools/live_topdown_view.py
```

Manual fallback:

```text
perception/tools/manual_topdown_view.py
```

Color quantization detector:

```text
perception/tools/color_quantization_detector.py
```

These scripts were moved under `perception/tools/`. Their imports have been
updated for the new layer layout. Until they are converted into proper package
entrypoints, run moved scripts from the repository root with `PYTHONPATH=.` if
they import other layer packages.

## Robot Origin Calibration

Robot-origin calibration is now under the Localization layer:

```text
localization/tools/robot_origin_calibration.py
```

It uses camera calibration, a top-down transform, and robot-mounted ArUco
markers to estimate robot-origin offsets written to:

```text
robot_calibration.json
```

The old top-down detector app no longer exists, so there is currently no
`topdown_object_detector.py --drive` calibration path.

## Legacy Main

```text
brain/Main.py
```

This is an early placeholder script, not the rebuilt Brain/FSM layer and not a
complete autonomous entrypoint.

## Tests

Camera/perspective tests currently live in `test/`:

```text
test/test_checkerboard_smoke.py
test/test_perspective_warp.py
test/checkerboard_smoke_test.py
test/perspective_warp_test.py
```

Run all current kept tests with:

```bash
env PYTHONPYCACHEPREFIX=/private/tmp/codex-pycache /Users/alex/miniforge3/bin/python3 -m unittest discover -s test -p 'test_*.py'
```
