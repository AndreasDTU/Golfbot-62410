# Changelog

All notable larger additions and behavioral changes to this repository should be recorded here.

## 2026-05-21

### Added

- Added ball-aware route planning for autonomous collection.
  - `pathfinding/planner.py` now inserts inflated hard obstacles for
    non-target balls during each selected-target planning attempt.
  - The selected target ball is excluded from its own obstacle layer so pickup
    standoff/final-pickup poses remain reachable.
  - Orange/VIP routing is absolute first. If a white ball blocks a safe orange
    route, the planner keeps orange as the selected target and may use an
    explicit `orange forced first` last-resort route instead of selecting a
    white intermediate pickup.
  - Non-target ball obstacle radius is computed from
    `ball_radius_cm + 0.5 * robot_width_cm + non_target_ball_extra_clearance_cm`.
  - Added route metadata for ball obstacles and ball-avoidance mode without
    adding schematic ball-avoidance overlays.
  - Added planner configuration defaults for enabling/disabling non-target ball
    avoidance, ball radius, small extra clearance, and last-resort orange
    contact.
- Added `tools/collector_playground.py`, a standalone open-loop collector
  playground for manually testing the GolfBot collector/pipe over the existing
  EV3 TCP controller without starting vision, route planning, ball detection,
  wheel dispatch, or autonomous driving.
  - The playground now starts an OpenCV HighGUI window by default, matching the
    existing `tools/pathfinding_sandbox.py` UI style, and keeps the original
    terminal REPL behind `--cli` / `--terminal`.
  - The GUI uses the same `CollectorPlayground.execute(...)` command handling
    as terminal mode, with host/port controls, bounded manual movement input,
    command logging, software-state display, and a simple open-loop robot/pipe
    visualization.
  - Added `--dummy` mode so the GUI or terminal REPL can be previewed without
    an EV3 hostname, robot server, or network connection.
  - `robot/controller.py` now exposes manual `pipe_up(...)`,
    `pipe_down(...)`, and `pipe_stop()` helpers plus configurable TCP port and
    timeout arguments.
  - `docs/COLLECTION_MECHANISM.md` documents the playground startup procedure,
    commands, confirmation behavior, and open-loop/no-sensor safety warning.
- Added explicit collection-actuator command separation.
  - `robot/controller.py` now exposes `collector_travel_position()` for safe driving height, `pickup_assist()` for collection, and `unload_full_cycle()` for goal unloading.
  - `robot/robot_server.py` maps `collector_travel_position` to a raise/travel command, `pickup_assist` to a small pipe jiggle, and `unload_full_cycle` to the full unloading stroke, while keeping `pickup`/`dropoff` as compatibility aliases.
  - `tools/topdown_object_detector.py` autonomous pickup state now calls only `pickup_assist()` and gates route following on a `TRAVEL` collector state.
  - Added `docs/COLLECTION_MECHANISM.md` to document the vertical tube, travel height, one-way retention, FIFO unloading, and actuator safety rule.
- Added step-by-step drive mode to `tools/topdown_object_detector.py`.
  - New `--step` flag works with `--drive` and starts robot motion in a paused operator-waiting state.
  - Pressing `n` releases one autonomous target run while vision, smoothing, occupancy-grid updates, and async route planning continue every frame.
  - The drive loop automatically pauses again after the existing pickup completion/replan transition.
  - Main debug output now shows a high-visibility paused prompt during step-mode waits.

### Verified

- Added focused tests for non-target ball obstacle insertion, selected target
  exclusion, configured ball obstacle inflation, route-around behavior,
  absolute orange-first selection, `orange forced first` fallback, white-ball
  avoidance after orange is absent, and debug disabling.
- Added focused tests for collection actuator command separation, collector travel-position gating, and autonomous white/orange collection assist behavior.
- Added focused tests for collector playground command dispatch, manual movement
  validation, unload confirmation, GUI-to-terminal command mapping, state
  transitions, dummy mode, startup safety, and isolation from the autonomous
  stack.
- Added focused tests for step-mode CLI parsing and pause/release behavior.

## 2026-05-20

### Added

- Added hybrid near-zone drive handoff for `tools/topdown_object_detector.py`.
  - UDP `LR <left> <right>` route tracking now halts at the configured near-zone boundary.
  - Final pickup approach uses calibrated TCP encoder commands with `turn(...)` followed by `move(...)`.
  - TCP command responses are returned from `robot/controller.py` so blocking movement calls can be observed reliably.
- Added edge-aware drive control in `robot/control.py`.
  - Computes robot body clearance to field edges using only the main body footprint.
  - Scales forward speed down and increases heading/XTE proportional gains near walls.
- Added pickup standoff planning in `pathfinding/planner.py`.
  - Generates valid pickup standoff/final-pickup pose pairs around each ball.
  - The final pickup pose places the tube center on the ball.
  - The standoff pose is exactly `15 cm` behind the final pickup pose along the same heading.
  - Hybrid A* now targets the standoff goal set dynamically instead of a single preselected pose.
- Added near-zone route visualization in `vision/debug.py`.
  - Route heatmap stops at the near-zone handoff point.
  - Red markers show UDP halt / TCP turn locations.
  - Final TCP center-to-center approach segment is drawn using the same speed heatmap palette as the UDP route.
  - Intermediate robot footprint clutter was removed; pickup footprints are shown only at collection poses.
- Added global ball-count reconciliation in `tools/topdown_object_detector.py`.
  - Drive dispatch waits for a stable initial YOLO ball count.
  - Successful TCP pickup moves optimistically increment `balls_collected`.
  - A 15-frame stable count debounce corrects missed/nudged pickups by decrementing `balls_collected`.
  - Debug overlay displays initial count, collected count, and stable visible count.

### Changed

- Rewrote `AGENTS.md` to match the current top-down object detection,
  Hybrid A*, and hybrid UDP/TCP robot-control architecture.
  - Updated the expected file structure.
  - Replaced outdated vision-system assumptions with the current
    `vision/`, `pathfinding/`, `robot/`, `tools/`, `docs/`, and `test/`
    module responsibilities.
  - Added current testing, safety, documentation, and changelog expectations.
- Updated route planning to better match differential-drive physical behavior.
  - Final TCP moves are constrained to be collinear with the robot heading.
  - The visual route now traces the robot body center only; it no longer draws route lines to ball/tube coordinates.
- Replaced the old blind open-loop pickup creep behavior with deterministic align-then-move TCP execution.
- Updated the closed-loop controller tests to cover:
  - edge slowdown/gain scaling,
  - near-zone UDP stop and TCP turn/move ordering,
  - route heatmap near-zone breaks,
  - pickup standoff geometry,
  - global ball-count reconciliation and debounce behavior.

### Verified

- Ran:

```text
env PYTHONPYCACHEPREFIX=/private/tmp/codex-pycache python3 -m unittest test.test_pathfinding_heuristic test.test_drive_control test.test_topdown_detector_app_shell
```

- Result: `Ran 19 tests ... OK`
