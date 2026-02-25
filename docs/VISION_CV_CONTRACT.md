# CV Vision Pipeline Contract --- Ping Pong Ball Collection Robot

Status: ACTIVE\
Owner: Team `<your team>`{=html}\
Scope: Real-time ball detection + mapping into a logical grid for
navigation commands.\
Non-scope: Training ML models (explicitly excluded).

## Goals

1.  Detect ping pong balls reliably in a contest arena with uneven
    lighting (sunlight + shade possible).
2.  Produce stable ball positions in a top-down coordinate system and
    map them to a logical grid.
3.  Run in real time (target: ≥ 20 FPS, stretch goal: 30 FPS) on the
    chosen compute device (laptop recommended).

## Hard Constraints

-   Environment lighting is not controllable (windows, shadows, mixed
    illumination).
-   Camera setup is controllable (mounting position, height, framing).
-   No model training. Pretrained "foundation segmentation" not used as
    core dependency.

------------------------------------------------------------------------

## System Outputs

Vision module outputs, every frame (or at fixed tick rate):

-   FrameId, Timestamp
-   ArenaPose: homography / warp transform used for top-down mapping
-   Balls: list of {x_world, y_world, confidence, radius_px, area_px}
-   GridOccupancy: logical grid with cell states:
    -   EMPTY, BALL, OBSTACLE, UNKNOWN (optional)
-   DebugOverlays (for development): masks + detections + grid + corner
    markers

------------------------------------------------------------------------

## Pipeline Overview (Required Order)

### 0) Camera Setup Requirements (physical)

-   Camera mounted rigidly, cannot move during run.
-   Prefer near top-down (minimize perspective skew).
-   Ensure all arena corners are visible (or visible corner markers are
    present).

### 1) Calibration & Geometry Normalization

Goal: Make spatial mapping consistent across the field.

1.  Lens calibration (once per camera/setup).
2.  Undistortion (every frame, fast remap).
3.  Perspective transform (homography) to a fixed top-down rectangle.

Invariant: After warp, the arena must be a stable rectangle with fixed
pixel dimensions.

### 2) Illumination Normalization

Goal: Reduce sensitivity to sunlight/shadows.

Required approach:

-   Convert warped frame → Lab
-   Apply CLAHE to L channel
-   Use a/b channels for color segmentation

Invariant: Detection must not collapse when half the field is in shade.

### 3) Segmentation (Ball Candidate Mask)

-   Create binary mask of likely ball pixels.
-   Apply morphology cleanup.

### 4) Candidate Extraction

-   Find connected components / contours.
-   Compute area, circularity, radius, bounding box.

### 5) Filtering & Scoring

Accept candidate if:

-   area within expected range
-   circularity above threshold
-   radius within expected range

Assign confidence score.

### 6) Tracking (optional)

-   Smooth detections across frames.
-   Reduce flicker.

### 7) Grid Mapping

Convert pixel coordinates to logical grid coordinates.

------------------------------------------------------------------------

## Performance Requirements

-   Minimum: ≥ 20 FPS
-   Stretch goal: 30 FPS
-   Max latency target: ≤ 50 ms per frame

------------------------------------------------------------------------

## Debug & Validation Requirements

Provide overlays:

-   Warped view
-   Segmentation mask
-   Detection circles
-   FPS display

Acceptance test:

Detection must work in bright, dark, and mixed lighting.

------------------------------------------------------------------------

# Risk Management

## Risk 1 --- Mixed illumination

Likelihood: High\
Impact: High

Mitigation:

-   Lab color space normalization
-   CLAHE normalization
-   Geometry filtering

Fallback:

-   Adaptive thresholding

------------------------------------------------------------------------

## Risk 2 --- Specular highlights

Likelihood: High\
Impact: Medium

Mitigation:

-   Morphological closing
-   Circle fitting

------------------------------------------------------------------------

## Risk 3 --- Camera auto-adjustment

Likelihood: Medium\
Impact: High

Mitigation:

-   Disable auto exposure if possible
-   Use Lab color channels

------------------------------------------------------------------------

## Risk 4 --- Homography failure

Likelihood: Medium\
Impact: High

Mitigation:

-   Manual corner calibration fallback

------------------------------------------------------------------------

## Risk 5 --- False positives

Likelihood: Medium\
Impact: Medium

Mitigation:

-   Geometry filters
-   Temporal filtering

------------------------------------------------------------------------

## Risk 6 --- Performance drop

Likelihood: Medium\
Impact: Medium

Mitigation:

-   Reduce resolution
-   Profile pipeline

------------------------------------------------------------------------

# Definition of Done

System is complete when:

-   Reliable detection in mixed lighting
-   ≥ 20 FPS achieved
-   Stable grid mapping
-   Debug tools functional
