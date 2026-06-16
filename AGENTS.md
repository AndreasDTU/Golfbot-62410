# AGENTS.md — Layered GolfBot Repository

Status: ACTIVE
Purpose: Define rules, contracts, and expectations for coding agents operating on this repository.
Scope: Entire repository unless overridden by a deeper `AGENTS.md`.

---

# System Summary

This repository is being reorganized around a layered ping-pong ball collection
robot architecture. The previous all-in-one autonomous detector app has been
removed during the movement rework. There is currently no complete autonomous
entrypoint equivalent to the deleted `tools/topdown_object_detector.py`.

Current layer folders:

```text
perception/    Kept camera, calibration, detection, tracking, mapping, and debug code
path/          Kept Hybrid A* route-planning code and path sandbox
localization/  Robot pose/localization code restored as reference for rebuild
brain/         New FSM layer target; not rebuilt yet
guidance/      Route-tracking/reference guidance code; not a clean new layer yet
control/       EV3 controller/server/tooling plus reference dispatch/telemetry code
docs/          Architecture, workflow, and rework documentation
test/          Remaining kept unit/smoke tests and assets
```

The movement rework status, layer audit, and updated stage plan are in:

```text
docs/MOVEMENT_REWORK_STATUS.md          — current source of truth
```

The original brainstorm proposals are kept for architectural intent but are
no longer current for status tracking:

```text
docs/refactor/movement_layer_architecture.pdf
docs/refactor/movement_rework_integration_plan.md
```

Grey layers from that architecture are kept code: Perception and Path. Green
layers are rebuild targets: Localization, Brain/FSM, Guidance, and Control.
Some green folders currently contain restored legacy/reference code; do not
mistake that for a completed clean-layer implementation.

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

## Perception

Perception code lives under `perception/`.

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
perception/camera/
perception/tools/
```

The YOLO model file is currently:

```text
perception/tools/best.pt
```

Do not replace the existing perception approach with a new detector family
unless explicitly asked.

The perception layer should continue to produce:

- top-down/warped frame data,
- red-zone detections and masks,
- raw ball detections,
- smoothed ball coordinates,
- occupancy grid,
- debug visualization data.

## Path

Path planning code lives under `path/`.

Important modules:

```text
path/pathfinding/models.py
path/pathfinding/planner.py
path/pathfinding/plancreation.py
path/tools/pathfinding_sandbox.py
```

The active kept route planner is Hybrid A* over:

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

## Localization, Brain, Guidance, And Control

These are movement-rework layers.

Current status:

```text
localization/localization.py       Restored legacy/reference localization
localization/models.py             Restored legacy/reference robot models
guidance/route_tracking.py         Restored legacy/reference route tracking
control/controller.py              TCP controller helper
control/io.py                      TCP wheel dispatch/reference commander
control/telemetry.py               Drive telemetry helper
control/robot/robot_server.py      EV3-side TCP command server
control/tools/drive_calibration.py Drive calibration helper
control/tools/collector_playground.py Manual collector playground
brain/Main.py                      Legacy placeholder, not a rebuilt Brain layer
```

The clean contracts required by the movement architecture are not complete yet:

- Control command API, sim/real backend, and boundary logging.
- Localization `RobotPose + valid/freshness` boundary.
- Guidance `intent + live pose -> turn/drive/adjust` boundary.
- Brain/FSM route cursor and arbitration layer.

Agents MUST NOT claim autonomous driving is currently available unless a new
entrypoint has actually been implemented and tested.

---

# File Structure Expectations

The current source structure is layer-based:

```text
brain/
control/
docs/
guidance/
localization/
path/
perception/
test/
```

Old root packages such as `vision/`, `camera/`, `pathfinding/`, `robot/`, and
`tools/` are not the current source locations. Do not add new code there unless
the user explicitly asks to restore that layout.

Imports should use current layer paths, for example:

```text
perception.vision.*
perception.camera.*
path.pathfinding.*
localization.*
guidance.route_tracking
control.*
```

---

# Modification Rules

Agents MUST:

- make minimal necessary changes,
- preserve existing module boundaries unless there is a clear reason,
- preserve debug tools and overlays in kept grey layers,
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
- silently modify calibration files, homography, or robot geometry defaults,
- reintroduce a fake autonomous shell just to satisfy old tests.

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

# Testing Requirements

Use the Miniforge Python runtime when dependencies are needed:

```bash
env PYTHONPYCACHEPREFIX=/private/tmp/codex-pycache /Users/alex/miniforge3/bin/python3 -m unittest discover -s test -p 'test_*.py'
```

Current remaining kept tests cover perception utilities, pathfinding, drive
calibration, telemetry, and the EV3 command server. Old autonomous app-shell and
drive-loop tests were removed with the deleted top-down app.

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

Graphify output is not currently present in this checkout. If `graphify-out/`
is restored later, use it as a navigation aid for architecture questions and
major structural changes.

---

# Safety & Failure Handling

If vision cannot confidently detect balls, the system must report no detections
or preserve safe prior state only when explicitly designed to do so.

If robot pose is missing, route is invalid, XTE is unsafe, or calibration is not
ready, the drive controller must stop or stay stopped.

All safety fallbacks should be deterministic, visible in debug/status overlays,
and easy to reason about during contest debugging.
