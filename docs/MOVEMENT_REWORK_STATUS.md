# Movement Rework Status

**Date:** 2026-06-15
**Version:** 1.0
**Purpose:** Single source of truth for the movement-layer rebuild. Supersedes the
brainstorm-stage documents in `docs/refactor/` for current status tracking.

Original brainstorm references (kept for architectural intent, no longer
current for status):

```text
docs/refactor/movement_layer_architecture.pdf   — 6-layer architecture proposal
docs/refactor/movement_rework_integration_plan.md — staged build & gate plan
```

---

## Layer Status Audit

### 1. Perception (Grey / Kept)

**Status:** Complete

**Key files:**

```text
perception/vision/config.py          Typed config (field, camera, detection, robot, planner, drive)
perception/vision/models.py          ParallaxConfig, shared vision models
perception/vision/calibration.py     Homography and ArUco calibration
perception/vision/preprocessing.py   Frame undistortion and warp
perception/vision/detection.py       YOLO + HSV ball and red-zone detection
perception/vision/tracking.py        Lucas-Kanade optical flow + EMA smoothing
perception/vision/grid_mapping.py    Occupancy grid from detections and red zones
perception/vision/pipeline.py        Per-frame pipeline orchestration
perception/vision/debug.py           Schematic and overlay rendering
perception/vision/geometry.py        CoordinateMapper, ParallaxCorrector
perception/camera/                   Camera capture, undistortion, colorspace utilities
perception/tools/                    Standalone calibration and debug tools; YOLO model (best.pt)
```

**What works:**

- YOLO ball detection with confidence/area filtering.
- HSV red-zone segmentation and masking.
- Lucas-Kanade optical-flow tracking with forward-backward error rejection
  and EMA coordinate smoothing.
- Parallax correction for elevated markers projected to the ground plane.
- Occupancy grid with red-zone hard obstacles and soft ball costmap layers.
- Full debug overlay pipeline (schematic, route heatmap, ball avoidance
  halos, robot footprint, pickup poses).
- Camera lens calibration, homography calibration, and live trackbar tuning.
- Focus-lock after ball-count stabilization.

**What's missing:** Nothing relative to the architecture proposal. Perception
is a kept grey layer and is not part of the movement rebuild.

**Boundary contract:** Perception hands downstream layers:

| Output | Type | Units |
|---|---|---|
| Warped top-down frame | `np.ndarray` (BGR) | pixels |
| Ball detections | list of `(x_cm, y_cm, label, track_id)` | cm, bottom-left origin |
| Smoothed ball positions | same, after LK+EMA | cm |
| Red-zone mask | `np.ndarray` (uint8 grid) | 1 cm cells |
| Occupancy grid | `np.ndarray` (uint8 hard + float32 cost) | 1 cm cells |
| Robot marker observations | `dict[int, RobotMarkerObservation]` | pixels + ground pixels |

**Tests:** `test_checkerboard_smoke.py` (4 tests), `test_perspective_warp.py`
(3 tests).

---

### 2. Path (Grey / Kept)

**Status:** Complete

**Key files:**

```text
path/pathfinding/models.py           HybridPose, RoutePlan, RouteTrackingError, RouteSegmentType
path/pathfinding/planner.py          Hybrid A*, Dijkstra heuristic, pickup standoff, soft costmap
path/pathfinding/plancreation.py     Async route cache, target selection, invalidation
path/tools/pathfinding_sandbox.py    Standalone route-planning sandbox
```

**What works:**

- Hybrid A* over `(x_cm, y_cm, theta_rad)` with weighted heuristic, gear
  shift penalty, steering change penalty, and reverse cost multiplier.
- Multi-target pickup with standoff/final-pickup pose pairs, tube offset,
  and body collision checking.
- Soft ball costmap (concentric cost bands) for non-target ball avoidance.
- Wall-aware pickup approach prioritization (wall-normal preference, diagonal
  fallback for tight corners).
- Route segment metadata (`TRANSIT`, `PIVOT`, `CREEP`) with intended speed
  percentages.
- Progressive fallback: full soft-cost search, then 10% costmap, then
  hard-only with relaxed heading tolerance.
- Greedy path pruning for straight-line shortcuts.
- Unload staging pose planning (perpendicular to wall, body-center offset).
- Async route cache with bucket-based invalidation.

**What's missing:** Nothing relative to the architecture proposal. Path is a
kept grey layer.

**Boundary contract:** Path receives occupancy grids and ball targets from
Perception, and produces:

| Output | Type | Units |
|---|---|---|
| Route points | `list[HybridPose]` | cm, rad, bottom-left |
| Segment types | `list[RouteSegmentType]` | enum |
| Segment speeds | `list[float]` | percent |
| Active target | `PlannedBallTarget` | cm |
| Pickup poses | `list[HybridPose]` | cm, rad |
| Ball obstacles | `list[PlannedBallTarget]` | cm |

**Tests:** `test_pathfinding_heuristic.py` (27 tests covering search,
costmap, standoff, corner balls, avoidance, fallback, unload routing).

---

### 3. Localization (Green / Rebuild Target)

**Status:** Partial — restored reference code is functional but not a clean
new-layer build.

**Key files:**

```text
localization/localization.py                    RobotMarkerDetector, RobotPoseEstimator, RobotCalibrationCollector
localization/models.py                          RobotPose, RobotGeometry, DriveRuntime, DriveControlState, WheelCommand
localization/tools/robot_origin_calibration.py  Standalone spin-calibration tool
```

**What works:**

- ArUco marker detection on the warped top-down frame.
- Parallax correction of marker corners from mounted height to ground plane.
- Multi-marker pose averaging (body origin + heading from two markers).
- Image yaw to field heading conversion.
- Robot origin calibration via spin-fit (min enclosing circle + ellipse
  quality check + JSON persistence).
- `RobotPose` dataclass with `x_cm`, `y_cm`, `heading_rad`, `tube_x_cm`,
  `tube_y_cm`.
- `RobotGeometry` with tunable body footprint and tube offset.
- `DriveRuntime` with state machine enum, commander reference, and route
  progress tracking.

**What's missing:**

- No freshness or validity flag on `RobotPose`. The architecture proposal
  requires a `valid` boolean and a `freshness` timestamp so downstream layers
  can distinguish "no detection" from "stale detection."
- No explicit latency measurement. The gate plan requires end-to-end
  localization latency to be measured and recorded.
- No boundary logging. Input (raw frame + marker detections) and output
  (pose) are not logged at the layer boundary.
- Tight coupling to `perception.vision.*` imports. The layer directly uses
  `HomographyCalibrator`, `CoordinateMapper`, `ParallaxCorrector`, and
  `ParallaxConfig` from perception rather than receiving them through a
  defined interface.
- `localization/models.py` contains drive-domain types (`DriveRuntime`,
  `DriveControlState`, `WheelCommand`) that belong in control or guidance,
  not localization. These were placed here during the restore to avoid
  circular imports.

**Boundary contract (current):**

| Direction | Data | Type |
|---|---|---|
| In | Warped BGR frame | `np.ndarray` |
| In | Live UI params + calibration dict | `dict` |
| Out | Robot pose (or `None`) | `RobotPose` |
| Out | Robot origin pixel position | `tuple[float, float]` |
| Out | Per-marker observations | `dict[int, RobotMarkerObservation]` |

**Tests:** None dedicated. Localization logic is exercised only through
integration with the (now deleted) top-down detector app.

---

### 4. Control (Green / Rebuild Target)

**Status:** Partial — TCP controller and EV3 server are production-tested, but
there is no abstracted command API or sim/real backend.

**Key files:**

```text
control/controller.py                TCP client (RobotController)
control/io.py                        TcpWheelDispatcher, MotorCommander
control/telemetry.py                 TelemetryFrame, DriveTelemetryRecorder, log_event
control/robot/robot_server.py        EV3-side TCP command server (ev3dev2)
control/robot/pc.py                  Interactive REPL for TCP commands
control/tools/drive_calibration.py   Calibration math, JSON persistence, response parsing
control/tools/collector_playground.py Standalone collector GUI/REPL
```

**What works:**

- TCP command protocol: `move`, `back`, `turn`, `LR`, `pipe up/down/stop`,
  `drivecal get/set`, `collector_travel_position`, `pickup_assist`,
  `unload_full_cycle`, `stop`, `ping`.
- `TcpWheelDispatcher` with finite-command validation, speed clipping, send
  interval throttling, deadband filtering, forced stops, and dispatch error
  reporting.
- `MotorCommander` with `steer()`, `tank_turn()`, `drive_straight()`,
  `stop()` — single sign-convention source.
- `DriveTelemetryRecorder` with per-frame ring buffer, CSV dump, and
  `log_event()` structured printing.
- Drive calibration: live 360-degree spin measurement, forward distance
  measurement, correction math, EV3-side JSON persistence of
  `axle_track_mm` and `mm_per_unit`.
- Collector playground for open-loop pipe testing without vision/planning.

**What's missing:**

- No sim/real backend. The architecture proposal requires a backend
  abstraction so the entire stack can run without the EV3. Currently,
  `RobotController` opens a real TCP socket unconditionally.
- No formal command API independent of transport. `RobotController` mixes
  TCP connection management with the command vocabulary.
- No boundary logging at the control layer input (commands received) and
  output (motor actions executed + measured results).
- Blocking vs ticking is resolved in practice — finite commands (`move`,
  `turn`, `back`) block until motor completion; continuous commands (`LR`)
  tick — but this is not documented as a formal contract.

**Boundary contract (current):**

| Direction | Data | Type |
|---|---|---|
| In (finite) | `move(d)`, `turn(theta)`, `back(d)` | blocking TCP string |
| In (continuous) | `LR left_pct right_pct` | non-blocking TCP string |
| In (actuator) | `pipe`, `collector_travel_position`, etc. | blocking TCP string |
| Out | `"ok: ..."` / `"error: ..."` | TCP string |

**Tests:** `test_robot_server_commands.py` (6 tests — blocking ack, LR,
drivecal get/set/reject), `test_drive_calibration.py` (6 tests — heading
unwrap, correction math, JSON persistence, response parsing),
`test_telemetry.py` (5 tests — ringbuffer, CSV dump, no-pose recording,
log_event format).

---

### 5. Guidance (Green / Rebuild Target)

**Status:** Partial — `route_tracking.py` contains a working PD route tracker
and safety guard, but it is reference code, not a clean "intent to command"
layer.

**Key files:**

```text
guidance/route_tracking.py    WheelCommandController, DriveSafetyGuard, route utilities
```

**What works:**

- `WheelCommandController`: PD controller with heading and cross-track error
  terms, derivative smoothing, edge-proximity speed scaling and gain
  boosting, goal-distance speed profiling (cruise/creep ramp), slew-rate
  limiting (acceleration + deceleration caps).
- `DriveSafetyGuard`: progressive (monotonic, windowed) route tracking to
  prevent path-intersection jumps, XTE guard with route-cache clear and
  replan trigger, master-controller step that dispatches through
  `MotorCommander`.
- Route checkpoint mapping, goal distance computation, and body edge
  clearance calculation.

**What's missing:**

- Not built as a clean "intent + live pose -> turn/drive/adjust" boundary.
  The current code operates directly on route point lists and tracking
  errors rather than on typed intents (`GoToWaypoint`, `PickupBall`,
  `Unload`, etc.).
- No formal Guidance contract or input/output specification.
- No boundary logging.
- Tight coupling: imports from both `localization.models` and
  `path.pathfinding.models`, and receives `DriveRuntime` (a localization
  model type) as mutable state.
- The architecture proposal's question about whether Guidance and Control
  are one layer or two is answered by practice: finite commands block, so
  the two-layer split holds, but Guidance currently reaches through to
  `MotorCommander` (a Control-layer type) rather than issuing abstract
  commands.

**Boundary contract (current):**

| Direction | Data | Type |
|---|---|---|
| In | Robot pose | `RobotPose` (from localization) |
| In | Route points | `list[HybridPose]` (from path) |
| In | DriveRuntime (mutable) | `DriveRuntime` |
| Out | Wheel command | `WheelCommand` (via `MotorCommander.steer`) |
| Out | Safety state transitions | `DriveControlState` enum |

**Tests:** None dedicated. The guidance logic was previously tested through
`test_drive_control.py` (now deleted with the autonomous app). The PD
controller and safety guard have no standalone test coverage.

---

### 6. Brain / FSM (Green / Rebuild Target)

**Status:** Stub — `brain/Main.py` is a legacy placeholder, not a rebuilt
Brain layer.

**Key files:**

```text
brain/Main.py    Legacy main loop — captures one camera frame, runs HSV processing, exits
```

**What works:** Nothing. The file imports `RobotController` and basic camera
utilities but does not implement an FSM, route cursor, mode arbitration, or
any autonomous behavior.

**What's missing:** Everything required by the architecture proposal:

- FSM with defined states (idle, follow, turn-to-waypoint, pickup, unload,
  recover, fault).
- Route-progress cursor that owns the current position along the active
  route.
- Mode arbitration between route following, special maneuvers (pickup,
  unload, pivot), and recovery from faults.
- Intent generation: the Brain should produce typed intents consumed by
  Guidance rather than directly commanding motors.
- Fault injection and recovery testing.
- Boundary logging of state transitions and intent outputs.

**Boundary contract:** None defined.

**Tests:** None.

---

## Units & Frames Contract

The following conventions are established and used consistently across
Perception, Path, Localization, and Guidance:

| Property | Convention | Pinned? |
|---|---|---|
| Linear unit | centimeters (`cm`) | Yes |
| Angular unit | radians (`rad`) | Yes |
| Field origin | bottom-left corner | Yes |
| X axis | positive rightward | Yes |
| Y axis | positive upward | Yes |
| Heading zero | along +X (east) | Yes |
| Positive rotation | counter-clockwise (CCW) | Yes |
| Field dimensions | 167.0 cm x 121.5 cm | Yes (in `FieldConfig`) |
| Grid resolution | 1 cm per cell | Yes |
| Occupancy grid origin | bottom-left, matching field | Yes |

**Remaining ambiguity:**

- The EV3 server uses `mm` internally for wheel diameter and axle track, and
  `degrees` for motor rotation. The `units_to_degrees()` conversion uses
  `MM_PER_UNIT` (calibrated). The exact physical meaning of "unit" in
  `move <units>` is millimeters-of-travel after applying `MM_PER_UNIT`,
  but this is not documented as a formal contract.
- Pipe motor units are arbitrary (`PIPE_DEGREES_PER_UNIT = 45.0`) and are
  not mapped to any physical distance.
- `DriveConfig` speeds are in `percent` (0-100 scale matching EV3
  `SpeedPercent`), not physical units.

---

## Cross-Cutting Gaps

### Boundary Logging

**Status:** Missing everywhere.

The integration plan requires every layer to log its timestamped input and
output from day one so that failures in a full run can be attributed to the
first boundary that diverged.

Currently, `control/telemetry.py` provides `log_event()` (structured print)
and `DriveTelemetryRecorder` (ring buffer + CSV dump), but these are used
only inside the drive dispatch path, not at layer boundaries. No other layer
has any logging infrastructure.

### Sim/Real Backend

**Status:** Missing.

The integration plan requires a sim/real backend behind the Control command
API so the whole stack can run without the EV3. No simulator, mock motor
backend, or hardware abstraction layer exists. `RobotController` connects to
a live TCP socket unconditionally. Tests that exercise the EV3 server
(`test_robot_server_commands.py`) mock the `ev3dev2` motor classes at import
time, but this is test scaffolding, not a reusable sim backend.

### Blocking vs. Ticking Commands

**Status:** Resolved in practice, not formally documented.

The open question from the brainstorm — "do commands block until done, or
tick with status feedback every frame?" — has been answered by the working
system:

- **Finite commands** (`move`, `turn`, `back`, `pipe`, `pickup_assist`,
  `unload_full_cycle`) **block** on the EV3 until motor completion, then
  return an `"ok: ..."` acknowledgment.
- **Continuous commands** (`LR`) **tick** — they set wheel speeds and return
  immediately; the caller sends new values each frame.

This hybrid model means Guidance and Control remain two layers (Guidance
issues blocking finite commands through `MotorCommander.tank_turn()` /
`drive_straight()` and ticking continuous commands through `steer()`). The
brainstorm's concern that blocking commands would collapse the two layers
does not apply because both patterns coexist.

---

## Updated Stage Plan

### Stage 1 — Control

**Current state:** The EV3 command server and TCP client are
production-tested across multiple contest sessions. Motor commands work
reliably. Drive calibration (spin measurement + forward measurement + JSON
persistence) is implemented and tested.

**Gate criteria — filled:**

- `turn(theta)`: works, accuracy measured via live drive calibration tool.
  Repeatability data exists in saved calibration JSON files.
- `move(d)`: works, accuracy measured via calibration tool.
- `LR` continuous drive: works, used for all route-following sessions.
- Command-to-motion latency: not formally measured or recorded as a number.

**Gate criteria — unfilled:**

- Formal accuracy table (`|measured - commanded| <= X` over N trials) not
  recorded as a spec document. The calibration tool produces correction
  factors, but the raw trial data and std-dev are not persisted.
- Sim/real backend not built.
- Boundary logging not wired.

**Verdict:** Control is functionally ready. The remaining gate items are
documentation/infrastructure, not capability gaps.

---

### Stage 1b — Localization

**Current state:** ArUco-based pose estimation works in the live detector.
Spin calibration tool produces validated `robot_calibration.json`.
Multi-marker averaging and parallax correction are implemented.

**Gate criteria — filled:**

- Static and dynamic pose estimation works with live camera.
- Pose tracks the robot during autonomous route following.

**Gate criteria — unfilled:**

- No `valid` / `freshness` fields on `RobotPose`.
- No formal accuracy measurement (static pose error, tracking error during
  motion) against ground truth.
- No end-to-end latency measurement.
- No boundary logging.
- Drive-domain types (`DriveRuntime`, `DriveControlState`, `WheelCommand`)
  are in `localization/models.py` and need to be relocated to avoid
  coupling localization to control/guidance concerns.
- No dedicated test suite.

**Verdict:** Localization is functional but needs the clean contract
(validity flag, latency measurement, type relocation) before Guidance can
be built against measured specs.

---

### Stage 2 — Guidance

**Current state:** `guidance/route_tracking.py` contains the PD controller
and safety guard that powered autonomous driving in the previous app. The
code works but was not built as a clean layer with an intent-based API.

**What exists:**

- PD route tracker with heading/XTE error, derivative terms, edge control,
  speed profiling, and slew limiting.
- Safety guard with progressive tracking, XTE limit, replan trigger.
- Route checkpoint and goal distance utilities.

**What needs to change:**

- Define the Guidance boundary: receives typed intents (from Brain) and
  live pose (from Localization), produces commands (to Control).
- Replace direct `MotorCommander` coupling with Control command API calls.
- Add boundary logging.
- Build against the measured Control accuracy and Localization latency from
  Stages 1 and 1b (when those specs are recorded).
- Write a dedicated test suite (the old `test_drive_control.py` was deleted
  with the autonomous app and needs to be re-created against the new
  contract).

**Verdict:** The core algorithm is proven. The rebuild is primarily about
boundary cleanup, not algorithm rewrite.

---

### Stage 3 — Brain / FSM

**Current state:** Empty stub. `brain/Main.py` is a legacy placeholder.

**Full build needed:**

- FSM with at least: IDLE, FOLLOW, PIVOT, PICKUP, UNLOAD, RECOVER, FAULT.
- Route-progress cursor (currently owned by `DriveRuntime` in guidance).
- Mode arbitration — the logic that decides when to switch from route
  following to a pickup maneuver, when to unload, when to recover from
  XTE violations, etc. This logic currently lives scattered in the deleted
  top-down detector app's drive loop.
- Intent generation for Guidance.
- Fault injection hooks for testing.
- Boundary logging of all state transitions and intent outputs.

**Verdict:** No code to salvage from `brain/Main.py`. The FSM logic exists
in the deleted app's drive loop and needs to be extracted and formalized.

---

### Stage 4 — Full Autonomous

**Current state:** No autonomous entrypoint exists. The previous
`tools/topdown_object_detector.py` was deleted during the cleanup.

**Integration target:**

- Path drives Brain with planned routes.
- Brain drives Guidance with intents.
- Guidance drives Control with commands.
- Perception feeds Localization and Path.
- All boundary logs active.
- End-to-end ball collection tested.

**Gate:** Collects N balls autonomously across M runs with success rate
>= X%. Every failure is attributable to a layer via boundary logs.

---

## What's Changed Since the Brainstorm

1. **Old autonomous app deleted.** `tools/topdown_object_detector.py` — the
   monolithic detector/driver/controller — was removed. There is no
   autonomous entrypoint.

2. **Code reorganized into layer folders.** Source moved from root
   `vision/`, `pathfinding/`, `robot/`, `tools/` into `perception/`,
   `path/`, `localization/`, `guidance/`, `control/`, `brain/`.

3. **Reference code restored.** Localization, guidance, and control folders
   contain restored legacy code, not clean new-layer implementations. The
   code works but does not satisfy the architecture proposal's boundary
   contracts.

4. **Blocking vs. ticking resolved.** Finite commands block; continuous
   commands tick. Both patterns coexist. Guidance and Control remain
   separate layers.

5. **Units contract pinned.** cm, radians, bottom-left origin, CCW
   positive. Used consistently across all layers.

6. **Drive calibration implemented.** Live 360-degree spin + forward
   measurement with EV3-side JSON persistence. Not present in the
   brainstorm.

7. **Telemetry infrastructure started.** Ring buffer, CSV dump, and
   structured event logging exist in control, but boundary logging is not
   wired at any layer interface.

8. **Perception and Path are stable.** Both grey layers received significant
   capability additions (soft ball costmap, wall-aware pickup, progressive
   fallback, visual servo, focus lock) and extensive test coverage since the
   brainstorm.

9. **Tests partially lost.** `test_drive_control.py` and
   `test_topdown_detector_app_shell.py` were deleted with the autonomous
   app. The guidance PD controller and the drive state machine have no
   current test coverage.

10. **AGENTS.md updated.** Now describes the layer folder structure and
    marks green layers as rebuild targets with incomplete contracts.
