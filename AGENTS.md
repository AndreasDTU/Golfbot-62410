# AGENTS.md — Layered GolfBot Repository

Status: ACTIVE
Purpose: Define rules, contracts, and expectations for coding agents operating on this repository.
Scope: Entire repository unless overridden by a deeper `AGENTS.md`.

---

# System Summary

This repository implements a layered ping-pong ball collection robot. A bottom-up
movement rework is in progress, following a staged integration plan where each
layer is proven on the real robot and frozen before the next one is built.

The main GUI entrypoint is `brain/Main.py` — a single-window OpenCV app showing
the live camera feed and 2D schematic side by side, with mode buttons for field
corner calibration and guidance isolation testing (Stage 2).

Current layer folders:

```text
perception/    Camera, calibration, detection, tracking, mapping, debug — KEPT, stable
path/          Hybrid A* route-planning and path sandbox — KEPT, stable
localization/  Robot pose estimation from ArUco markers — rebuilt, functional
guidance/      Waypoint follower (GuidanceController) + legacy route tracking — Stage 2 active
control/       RobotCommander (TCP→EV3), telemetry, tools — Stage 1 complete
brain/         Main GUI with guidance test mode; FSM not yet built — Stage 3 target
docs/          Architecture, workflow, and rework documentation
test/          Unit tests for commander, guidance, pathfinding, telemetry, robot server
```

## Stage completion status

| Stage | Layer | Status |
|-------|-------|--------|
| 1 | **Control** (`control/commander.py`) | Built. `RobotCommander` provides `turn`/`drive`/`adjust`/`stop` over TCP. Unit-tested. Movement playground available for manual testing. |
| 1b | **Localization** (`localization/`) | Functional. `RobotPoseEstimator` produces `RobotPose` from ArUco detections. Used by Main GUI. |
| 2 | **Guidance** (`guidance/guidance.py`) | Built. `GuidanceController` is a per-frame waypoint follower with turn/drive hysteresis. Unit-tested. **Currently under isolation testing on the real robot** via Main GUI's Guide Test mode with hardcoded routes. |
| 3 | **Brain/FSM** | Not started. Will own route-progress cursor and mode arbitration. |
| 4 | **Full autonomous** | Not started. Path layer drives the Brain. |

The staged integration plan is in:

```text
docs/refactor/movement_rework_integration_plan.md
```

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

## Perception

Perception code lives under `perception/`. This layer is **stable and kept** —
do not restructure without explicit instruction.

Important modules:

```text
perception/vision/config.py           All configuration dataclasses (AppConfig, DriveConfig, etc.)
perception/vision/pipeline.py         VisionPipeline: frame → detections
perception/vision/preprocessing.py    Undistortion, homography warp (FramePreprocessor)
perception/vision/calibration.py      HomographyCalibrator (ArUco + manual corner modes)
perception/vision/detection.py        YOLO ball detection
perception/vision/tracking.py         Ball tracking / smoothing
perception/vision/grid_mapping.py     Occupancy grid
perception/vision/geometry.py         CoordinateMapper (pixel ↔ cm ↔ schematic)
perception/vision/debug.py            DebugRenderer (camera overlays, schematic, route viz)
perception/vision/models.py           Vision data models
perception/camera/                    Camera utilities
perception/tools/                     Standalone calibration/visualization tools
perception/tools/best.pt              YOLO model weights
```

The perception layer produces: top-down/warped frames, red-zone detections,
raw and smoothed ball coordinates, occupancy grid, and debug visualization.

Do not replace the existing perception approach with a new detector family
unless explicitly asked.

## Path

Path planning code lives under `path/`. This layer is **stable and kept**.

```text
path/pathfinding/models.py            HybridPose, PlannedBallTarget, RouteSegmentType
path/pathfinding/planner.py           HybridAStarPlanner, GreedyRoutePlanner
path/pathfinding/plancreation.py      Plan creation utilities
```

Routes are body-center trajectories in `(x_cm, y_cm, theta_rad)`. The planner
accounts for robot body footprint, pickup tube offset, and differential-drive
movement limits.

## Control (Stage 1 — complete)

```text
control/commander.py                  RobotCommander: unified TCP transport + movement API
control/telemetry.py                  Drive telemetry ring buffer + logging
control/robot/robot_server.py         EV3-side TCP command server
control/tools/movement_playground.py  Manual keyboard teleop for testing Control in isolation
control/tools/collector_playground.py Manual collector/actuator testing
control/tools/drive_calibration.py    Drive calibration helper
```

`RobotCommander` is the **single control surface** for all EV3 motor commands:

- `turn(degrees)` — in-place rotation, speed profiled by remaining angle
- `drive(cm, dt_s)` — forward/backward, speed profiled by distance, slew-rate limited
- `adjust(degrees)` — arc correction reusing the last drive speed
- `stop()` — zero wheel speeds

All movement commands are **non-blocking** and produce a single `LR` wheel-speed
message per call. Speed profiling, slew-rate limiting, rate limiting, and
deadband filtering are internal to the commander.

`controller.py` and `io.py` have been **deleted** — their functionality was
consolidated into `commander.py`.

## Localization (Stage 1b — functional)

```text
localization/localization.py          RobotPoseEstimator, RobotCalibrationCollector, normalize_angle
localization/models.py                RobotPose, RobotMarkerObservation, etc.
```

`RobotPoseEstimator.estimate()` takes a top-down frame and returns
`RobotPose(x_cm, y_cm, heading_rad, tube_x_cm, tube_y_cm)` or `None` if
markers are not visible.

## Guidance (Stage 2 — under isolation testing)

```text
guidance/guidance.py                  GuidanceController: per-frame waypoint follower (NEW)
guidance/route_tracking.py            Legacy WheelCommandController + PD route tracking (DEPRECATED)
```

`GuidanceController` is the **clean Stage 2 implementation**. It uses
`RobotCommander` exclusively — no direct wheel-speed or legacy `steer()` calls.

Key behavior:
- `set_route(list[HybridPose])` / `clear_route()`
- `tick(pose, dt_s) → GuidanceStatus` (RUNNING / ARRIVED / NO_POSE / NO_ROUTE / ERROR)
- Turn/drive hysteresis: enters turn mode when heading error > 70°, stays in
  turn until error < 10° (configurable via `turn_complete_threshold_rad`)
- Waypoint arrival tolerance: 4 cm (configurable)

`route_tracking.py` is legacy code kept for reference and debug visualization.
New code must not use `WheelCommandController` or the `steer()` API.

## Brain / Main GUI (Stage 3 — not started)

```text
brain/Main.py                         Main GUI: camera + schematic + guidance test mode
```

`brain/Main.py` is the **current operational entrypoint**. It provides:

- Live camera feed (left panel) with top-down warp after corner calibration
- 2D schematic field view (right panel) with ball, robot, and route visualization
- Field corner calibration via a temporary selection window (`Set Corners` / `f`)
- Guidance isolation testing (`Guide Test` / `g`): connects to the robot,
  loads hardcoded test routes (straight, 90-turn, L-shape), and runs
  `GuidanceController.tick()` each frame with the live pose
- Route cycling: press `g` while in guidance test mode to switch routes
- Clean stop/disconnect: `Stop` / `s` clears the route and disconnects

The Brain/FSM layer (route cursor, mode arbitration, fault recovery) has **not
been built yet**. When it is, it will sit between the Path layer and Guidance.

---

# Configuration

All configuration lives in `perception/vision/config.py` as frozen dataclasses:

```text
AppConfig           Top-level bundle (paths, field, camera, windows, robot, detection, planner, drive)
DriveConfig         Motor speeds, TCP settings, speed profiles, control gains
FieldConfig         Physical field dimensions
CameraConfig        Camera index, resolution, warp size
PlannerConfig       Hybrid A* planner parameters
```

`DriveConfig` contains the primary motor speed tuning knobs:

- `turn_speed_pct` / `turn_creep_speed_pct` — in-place rotation speed profile
- `base_speed_pct` / `creep_speed_pct` — forward drive speed profile
- `adjust_gain` — arc correction proportional gain
- `max_heading_for_forward_rad` — threshold to enter turn mode (70°)
- `turn_complete_threshold_rad` — threshold to exit turn mode (10°)

`DriveConfig` is the **single source of truth** for speed parameters. Both
`RobotCommander` and `GuidanceController` read from the same config instance.

---

# Units and Frames Contract

- **Distance**: centimeters (cm) everywhere
- **Heading**: radians, 0 = +X axis, positive = CCW, normalized to [-π, π)
- **Commander API**: `turn()` and `adjust()` take **degrees**; `drive()` takes **cm**
- **Guidance converts** at the call site with `math.degrees()`
- **Wheel speeds**: percentage of max (-100 to +100), `left = base - turn`, `right = base + turn`
- **Field origin**: top-left corner of the physical field

---

# File Structure

```text
brain/                  Main GUI entrypoint
control/                RobotCommander, telemetry, EV3 server, tools
  commander.py          Unified movement API (replaces deleted controller.py + io.py)
  telemetry.py          Boundary logging
  robot/                EV3-side server
  tools/                Movement playground, collector playground, calibration
docs/                   Architecture and rework documentation
guidance/               GuidanceController (new) + legacy route tracking
localization/           Robot pose estimation
path/                   Hybrid A* route planning
perception/             Vision pipeline, detection, calibration, debug
test/                   Unit and smoke tests
```

Imports use current layer paths:

```text
control.commander
control.telemetry
guidance.guidance
guidance.route_tracking     (legacy, for debug viz only)
localization.localization
localization.models
path.pathfinding.models
path.pathfinding.planner
perception.vision.*
perception.camera.*
```

---

# Modification Rules

Agents MUST:

- make minimal necessary changes,
- preserve existing module boundaries unless there is a clear reason,
- preserve debug tools and overlays in kept layers,
- keep data flow explicit,
- keep behavior deterministic,
- avoid hidden global state,
- avoid unnecessary per-frame allocations in live loops,
- precompute or cache static transforms where practical,
- update tests for changed behavior,
- respect the freeze-on-pass principle: do not modify a frozen layer without re-running its gate.

Agents MUST NOT:

- rewrite large systems without explicit instruction,
- change external interfaces or output formats casually,
- remove safety checks,
- remove calibration/debug tools,
- introduce training pipelines without approval,
- silently modify calibration files, homography, or robot geometry defaults,
- bypass `RobotCommander` with direct wheel-speed or `steer()` calls in new code,
- claim autonomous driving is available (Stage 4 is not complete).

External dependencies should remain conservative. OpenCV, NumPy, and standard
Python libraries are already core dependencies. New runtime dependencies require
clear justification and user approval.

---

# Performance Rules

The final detector and drive loop are intended for real-time operation.

Target:

- minimum sustained vision/control rate: 20 FPS,
- stretch target: 30 FPS,
- low-latency route/control updates.

Route planning may be more expensive than detection, but it should remain
cached/asynchronous where practical and should not block every frame in a final
live loop.

---

# Testing

Run tests with:

```bash
python -m pytest test/ -v
```

Current test coverage:

```text
test/test_commander.py              RobotCommander: turn/drive/adjust profiling, rate limiting, validation
test/test_guidance.py               GuidanceController: turn hysteresis, arrival, route management
test/test_pathfinding_heuristic.py  Hybrid A* planner heuristics and costs
test/test_drive_calibration.py      Drive calibration parsing
test/test_telemetry.py              Telemetry ring buffer
test/test_robot_server_commands.py  EV3 command server parsing
test/test_checkerboard_smoke.py     Checkerboard calibration smoke test
test/test_perspective_warp.py       Perspective warp utilities
```

For docs-only changes, tests are optional, but agents should state that tests
were not run because the change was documentation-only.

---

# Documentation & Changelog

Agents MUST update `CHANGELOG.md` whenever making a larger addition,
behavioral change, architecture change, control change, or user-visible
workflow change.

Agents SHOULD also update relevant docs when architecture or behavior changes:

- `docs/MOVEMENT_REWORK_STATUS.md`
- `docs/PATHFINDING_ARCHITECTURE.md`
- `docs/VISION_CV_CONTRACT.md`
- `docs/CAMERA_WORKFLOW_OVERVIEW.md`
- `docs/COLLECTION_MECHANISM.md`
- `docs/AUTONOMOUS_DRIVE_QUICKSTART.md`

---

# Safety & Failure Handling

If vision cannot confidently detect balls, the system must report no detections
or preserve safe prior state only when explicitly designed to do so.

If robot pose is missing, route is invalid, XTE is unsafe, or calibration is not
ready, the drive controller must stop or stay stopped.

When guidance receives `None` pose, it immediately sends `stop()` and returns
`NO_POSE`. When no route is loaded, it returns `NO_ROUTE` without sending any
commands.

All safety fallbacks should be deterministic, visible in debug/status overlays,
and easy to reason about during contest debugging.
