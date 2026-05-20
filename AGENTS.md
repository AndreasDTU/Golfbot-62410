# AGENTS.md — Top-Down Detection, Routing & Robot Control

Status: ACTIVE
Purpose: Define rules, contracts, and expectations for coding agents operating on this repository.
Scope: Entire repository unless overridden by a deeper `AGENTS.md`.

---

# System Summary

This repository implements a contest-critical ping-pong ball collection system.

The current main application is:

```text
tools/topdown_object_detector.py
```

It owns the OpenCV UI shell, live/video/image execution modes, service wiring,
route planning requests, robot pose updates, drive-state management, and
keyboard dispatch.

Domain behavior is split into:

```text
vision/       Camera calibration, preprocessing, detection, tracking, mapping, debug rendering
pathfinding/ Hybrid A* route planning, robot-footprint collision checking, route models
robot/        Robot localization, drive control, UDP wheel dispatch, TCP controller helpers
tools/        Executable calibration, visualization, sandbox, and detector utilities
docs/         Architecture, workflow, test, and pipeline documentation
test/         Unit/smoke tests and test assets
```

Agents must treat this as a real-time robotics system where reliability,
determinism, debuggability, and safe failure behavior matter more than elegance.

---

# Priority Order

When making decisions, agents MUST prioritize:

1. Contract correctness
2. Contest reliability
3. Robot safety
4. Determinism
5. Debuggability
6. Real-time performance
7. Code elegance

Undefined or unsafe behavior is unacceptable. When unsure, choose the safest
deterministic option.

---

# Current Architecture

## Vision Pipeline

The extracted vision stack lives in `vision/`.

Important modules:

```text
vision/calibration.py     Homography and camera calibration helpers
vision/preprocessing.py   Undistortion, perspective transform, optional normalization
vision/detection.py       Red-zone and ball detection
vision/tracking.py        Ball smoothing/tracking
vision/grid_mapping.py    Occupancy-grid construction
vision/pipeline.py        Vision pipeline orchestration
vision/debug.py           Debug overlays, schematic view, route heatmap
vision/config.py          Typed configuration values
vision/models.py          Vision data models
vision/geometry.py        Coordinate mapping and geometry helpers
```

The detector currently uses YOLO-backed ball detection through the existing
local model file in `tools/best.pt`, plus classical OpenCV geometry,
calibration, grid mapping, and debug rendering. Do not replace this stack with a
new detector family unless explicitly asked.

The vision pipeline should continue to produce:

- top-down/warped frame data,
- red-zone detections and masks,
- raw ball detections,
- smoothed ball coordinates,
- occupancy grid,
- debug visualization data.

Agents MUST preserve debug visibility for:

- top-down/warped view,
- segmentation masks,
- ball/red-zone detections,
- robot pose and calibration status,
- route/occupancy visualization,
- drive state, XTE/heading error, FPS, and timing/status overlays.

## Pathfinding

Routing lives in `pathfinding/`.

Important modules:

```text
pathfinding/models.py   HybridPose, PlannedBallTarget, RoutePlan, RouteTrackingError
pathfinding/planner.py  Hybrid A*, collision checking, route facade, greedy collection route
```

The active route planner is Hybrid A* over:

```text
x_cm, y_cm, theta_rad
```

Routes are robot body-center trajectories. The planner must account for robot
body footprint, pickup tube offset, and differential-drive movement limits.

Pickup planning uses valid standoff/final-pickup pose pairs:

- the final pickup pose places the tube center on the ball,
- the standoff pose is translated backward along the same heading by the
  near-zone distance,
- the final TCP segment must be collinear with robot heading,
- the main robot body must remain collision-free,
- the pickup tube may overhang field walls when appropriate.

Do not simplify pickup planning back to raw ball coordinates or a single
hardcoded approach pose.

## Robot Localization & Control

Robot-domain code lives in `robot/`.

Important modules:

```text
robot/localization.py  ArUco-based robot pose estimation and calibration collection
robot/control.py       Route tracking, XTE/heading control, edge-aware wheel command computation
robot/io.py            Non-blocking UDP left/right wheel-speed dispatch
robot/controller.py    TCP command helper for calibrated encoder actions
robot/models.py        Robot geometry, pose, command, runtime state models
robot/robot_server.py  EV3-side TCP command server
```

Closed-loop driving under `--drive` uses a hybrid control architecture:

- UDP `LR <left> <right>` commands for continuous route tracking.
- TCP `turn(...)` and `move(...)` commands for calibrated near-zone pickup.

Agents MUST ensure robot commands are finite and validated. The robot must
never receive NaN, undefined, random, or unvalidated coordinates/speeds.

If pose, route, calibration, or detection is invalid, allowed behavior is:

- send/keep zero wheel speed,
- clear or preserve route cache as appropriate,
- return empty detections,
- replan from validated state.

Not allowed:

- fabricate detections,
- fabricate robot position,
- silently recalibrate during operation,
- silently mutate homography during drive.

---

# File Structure Expectations

The current expected structure is:

```text
AGENTS.md
CHANGELOG.md

docs/
    CAMERA_WORKFLOW_OVERVIEW.md
    PATHFINDING_ARCHITECTURE.md
    TEST_OVERVIEW.md
    VISION_CV_CONTRACT.md

vision/
    __init__.py
    calibration.py
    config.py
    debug.py
    detection.py
    geometry.py
    grid_mapping.py
    models.py
    pipeline.py
    preprocessing.py
    tracking.py

pathfinding/
    models.py
    planner.py
    plancreation.py

robot/
    calibrate.py
    control.py
    controller.py
    io.py
    localization.py
    models.py
    pc.py
    robot_server.py

tools/
    auto_topdown_aruco.py
    calibrate_camera.py
    checkerboard_smoke_test.py
    color_quantization_detector.py
    live_topdown_view.py
    live_undistort.py
    manual_topdown_view.py
    pathfinding_sandbox.py
    perspective_warp_test.py
    robot_origin_calibration.py
    topdown_object_detector.py
    undistort_bane.py

test/
    test_checkerboard_smoke.py
    test_drive_control.py
    test_pathfinding_heuristic.py
    test_perspective_warp.py
    test_topdown_detector_app_shell.py
```

Do not force the repository back into older placeholder structures. Add new
files where they fit the current architecture.

---

# Modification Rules

Agents MUST:

- make minimal necessary changes,
- preserve existing module boundaries unless there is a clear reason,
- preserve debug tools and overlays,
- keep data flow explicit,
- keep behavior deterministic,
- avoid hidden global state,
- avoid unnecessary per-frame allocations in live loops,
- precompute or cache static transforms where practical,
- update tests for changed behavior.

Agents MUST NOT:

- rewrite large systems without explicit instruction,
- change external interfaces or output formats casually,
- remove safety checks,
- remove calibration/debug tools,
- introduce training pipelines without approval,
- silently modify calibration files, homography, or robot geometry defaults.

External dependencies should remain conservative. OpenCV, NumPy, and standard
Python libraries are already core dependencies. New runtime dependencies require
clear justification and user approval.

---

# Performance Rules

The detector and drive loop are intended for real-time operation.

Target:

- minimum sustained vision/control rate: 20 FPS,
- stretch target: 30 FPS,
- low-latency route/control updates.

Agents SHOULD profile or reason explicitly before adding heavy per-frame work.
Avoid expensive initialization inside the frame loop.

Route planning may be more expensive than detection, but it should remain
cached/asynchronous where possible and should not block every frame.

---

# Testing Requirements

Agents MUST validate new behavior with relevant tests whenever practical.

Common focused test commands:

```text
env PYTHONPYCACHEPREFIX=/private/tmp/codex-pycache python3 -m unittest test.test_drive_control
env PYTHONPYCACHEPREFIX=/private/tmp/codex-pycache python3 -m unittest test.test_pathfinding_heuristic
env PYTHONPYCACHEPREFIX=/private/tmp/codex-pycache python3 -m unittest test.test_topdown_detector_app_shell
env PYTHONPYCACHEPREFIX=/private/tmp/codex-pycache python3 -m unittest test.test_perspective_warp test.test_checkerboard_smoke
```

Use the `PYTHONPYCACHEPREFIX` form when the default Python cache path is outside
the writable workspace.

For docs-only changes, tests are optional, but agents should state that tests
were not run because the change was documentation-only.

---

# Documentation & Changelog

Agents MUST update `CHANGELOG.md` whenever making a larger addition, behavioral
change, architecture change, control change, or user-visible workflow change.

Changelog entries SHOULD summarize:

- what was added or changed,
- why the change matters operationally,
- which key files or subsystems were affected,
- what validation or tests were run.

Tiny typo fixes or purely local cleanup do not require a changelog entry.

Agents SHOULD also update relevant docs when architecture or behavior changes:

- `docs/PATHFINDING_ARCHITECTURE.md` for route planning, drive control, pickup
  standoff, route visualization, and reconciliation behavior.
- `docs/VISION_CV_CONTRACT.md` for vision pipeline contracts.
- `docs/CAMERA_WORKFLOW_OVERVIEW.md` for calibration/camera workflow.
- `docs/TEST_OVERVIEW.md` for test workflow changes.

---

# Safety & Failure Handling

If vision cannot confidently detect balls, the system must report no detections
or preserve safe prior state only when explicitly designed to do so.

If robot pose is missing, route is invalid, XTE is unsafe, or calibration is not
ready, the drive controller must stop or stay stopped.

If pickup reconciliation detects a likely missed/nudged ball, it should correct
the optimistic pickup count so the existing detector and planner can naturally
route to the visible ball again.

All safety fallbacks should be deterministic, visible in debug/status overlays,
and easy to reason about during contest debugging.

---

# Current Documentation

Before making larger changes, read the relevant docs:

- `docs/VISION_CV_CONTRACT.md`
- `docs/PATHFINDING_ARCHITECTURE.md`
- `docs/CAMERA_WORKFLOW_OVERVIEW.md`
- `docs/TEST_OVERVIEW.md`
- `CHANGELOG.md`

These documents are part of the working contract. Update them when the contract
or behavior changes.
