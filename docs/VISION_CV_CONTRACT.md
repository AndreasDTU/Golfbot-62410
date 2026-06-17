# CV Vision Pipeline Contract

Status: ACTIVE KEPT LAYER

Scope: Perception code for camera calibration, top-down preprocessing,
ball/red-zone detection, tracking, coordinate mapping, occupancy-grid
construction, and debug rendering.

Non-scope: training new ML models or replacing the detector family.

## Current Locations

```text
perception/vision/
perception/camera/
perception/tools/
```

Important modules:

```text
perception/vision/calibration.py
perception/vision/preprocessing.py
perception/vision/detection.py
perception/vision/tracking.py
perception/vision/grid_mapping.py
perception/vision/pipeline.py
perception/vision/debug.py
perception/vision/config.py
perception/vision/models.py
perception/vision/geometry.py
```

The YOLO model file currently lives at:

```text
perception/tools/best.pt
```

## Current Boundary

Perception may produce:

- warped/top-down frame data,
- red-zone detections and masks,
- raw ball detections,
- smoothed ball field coordinates,
- occupancy grid data,
- debug visualization data.

Path planning consumes mapped field coordinates and occupancy information after
grid mapping. The previous all-in-one detector UI has been deleted, so there is
currently no complete live app that wires this layer to Brain/Guidance/Control.

## Required Pipeline Order

1. Lens calibration for a camera/setup.
2. Fast undistortion for frames that use that calibration.
3. Top-down perspective transform or homography.
4. Optional normalization/preprocessing.
5. Ball and red-zone detection.
6. Tracking/smoothing.
7. Field-coordinate and occupancy-grid mapping.
8. Debug rendering.

## Debug Requirements

Preserve visibility for:

- warped/top-down view,
- segmentation masks,
- ball/red-zone detections,
- robot pose and calibration status when supplied by Localization,
- route/occupancy visualization when supplied by Path,
- timing/FPS overlays in future app shells.

## Performance Target

The final live system should sustain at least 20 FPS, with 30 FPS as the stretch
target. Avoid expensive per-frame initialization in this kept layer.

## Validation

Current perception-adjacent tests are run through:

```bash
env PYTHONPYCACHEPREFIX=/private/tmp/codex-pycache /Users/alex/miniforge3/bin/python3 -m unittest discover -s test -p 'test_*.py'
```
