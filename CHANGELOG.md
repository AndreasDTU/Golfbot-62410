# Changelog

All notable larger additions and behavioral changes to this repository should be recorded here.

## 2026-06-10

### Added

- Added precision-seeking visual-servo alignment before the final near-zone TCP
  pickup move.
  - The robot now performs the initial TCP turn, then consumes fresh camera
    pose/ball observations frame-by-frame while issuing proportional TCP
    micro-turns.
  - Alignment continues until the measured lateral error reaches the configured
    camera noise floor, stops improving, or hits a bounded iteration limit.

### Changed

- Corrected near-zone pickup alignment and cached-route pickup consumption.
  - Initial pickup targeting now derives the tube-offset body target from the
    intended line-of-sight approach to the observed ball centroid instead of the
    robot's stale pre-turn heading.
  - Successful cached-route pickup completion now consumes the completed pickup
    checkpoint and clears stale active-target metadata before returning to
    navigation, preventing immediate re-entry into the same pickup.

## 2026-06-08

### Added

- Added state-driven route continuation and unload routing for autonomous
  collection.
  - Pickup completion now keeps the cached global route when the visible
    remaining balls are still a stationary subset of the original plan.
  - Cached routes are invalidated only when remaining balls move across the
    configured target bucket, a new/hidden ball appears, robot drift exceeds
    route tolerance, or geometry metadata changes.
  - Empty visible-ball frames with `balls_collected > 0` now submit an explicit
    direct-to-unload route instead of waiting for ball targets.
  - The route planner treats an empty target list as a direct unload request
    when invoked by that state.

### Changed

- Removed automatic autonomous `collector_travel_position()` gating from the
  live drive loop. The collector safe position is now an operator precondition;
  the command remains available for manual use.

- Switched the autonomous `--drive` command transport to TCP-only.
  - Replaced the PC-side wheel dispatcher with a TCP wheel dispatcher that
    preserves finite-command validation, clipping, send interval/deadband
    filtering, forced stops, and dispatch error reporting.
  - Added TCP `LR <left> <right>` handling to `robot/robot_server.py` for
    continuous route-following wheel speeds on the same command server used by
    `move`, `turn`, and collector commands.
  - Updated drive wiring, operator docs, and tests so the old hybrid transport
    scenario is no longer part of `--drive`.

## 2026-05-28

### Added

- Added schematic debug overlays for soft non-target ball avoidance.
  - `vision/debug.py` now draws faint yellow avoidance halos from
    `RoutePlan.ball_obstacles` and `ball_obstacle_radius_cm`.
  - Planned route samples that enter a ball avoidance radius mark that ball with
    a red warning ring and `COLLISION` label.
  - Collision checks now evaluate the route chronologically and remove
    intentional pickup balls only after their own tube-center pickup segment is
    reached, so future targets remain active obstacles until their segment.
  - Collision detection checks full route line segments against avoidance
    circles instead of only sampled route points, catching sparse-waypoint
    drive-throughs.
  - The live detector and pathfinding sandbox now pass route ball-avoidance
    metadata into the schematic renderer.

## 2026-05-24

### Added

- Added a minimal autonomous terminal unload sequence after the robot reaches
  the planned unload endpoint.
  - The drive loop stops wheel output, preserves the final unload route
    after the last optimistic pickup, and runs a pipe-only double unload:
    `unload_full_cycle`, configurable `pipe_down`/`pipe_up` shake cycles,
    second `unload_full_cycle`, then `pipe_stop`.
  - The shake uses only existing pipe motor commands; no wheel `move`/`back`
    commands are used for shaking.
  - Added configurable pipe-shake units, speed, cycle count, and unload trigger
    distance in `DriveConfig`.

### Changed

- Tightened autonomous drive and unload safety gates before live contest runs.
  - EV3 TCP `move`/`turn` commands now acknowledge only after motor completion
    instead of allowing the subsequent TCP command to overlap.
  - `RobotController` no longer recursively replays actuator commands after a
    possible send; TCP command failures now surface as bounded `RuntimeError`s.
  - Manual wheel movement keys are ignored while pickup/unload special actions
    own control, while the manual stop key remains active.
  - Terminal unload now requires the optimistic collection count to be complete
    and async route results are rejected if the robot start pose bucket changed.

- Replaced hard non-target ball obstacles with a soft ball costmap.
  - `pathfinding/planner.py` now keeps the red-zone grid as the hard `uint8`
    obstacle layer and builds a parallel `float32` costmap for non-target balls.
  - Hybrid A* adds the sampled ball-cost value into `g(n)`, and
    `GridDijkstraHeuristic` includes the same cost layer in its 2D expansion.
  - Non-target balls now use concentric cost bands: core `1000.0`, close
    `200.0`, and warning `50.0`.
  - Removed the old `orange forced first` / `ball contact fallback` replanning
    pass; a single soft-cost search can still cross a ball region if that is the
    only validated route through hard obstacles.
  - `docs/PATHFINDING_ARCHITECTURE.md` documents the costmap behavior.

- Tuned Hybrid A* search to reduce soft-cost planning latency and zigzag routes.
  - Added `heuristic_weight = 1.5` for weighted A* priority scoring.
  - Added `gear_shift_penalty = 50.0` when a route switches between forward and
    reverse motion primitives.
  - Added `steering_change_penalty = 3.0`, increased
    `reverse_cost_multiplier` to `2.5`, and set `in_place_rotation_cost` to
    `2.0` to favor pivot-and-drive paths without making pivots prohibitively
    expensive.
  - Pickup standoff planning now accepts already-close, aligned, line-of-sight
    handoff poses instead of forcing an exact 15 cm standoff point.
  - Added greedy path pruning for collision-free straight-line shortcuts that do
    not enter worse soft-cost bands.

- Protected the terminal pickup maneuver from path pruning.
  - Hybrid A* now appends the final `standoff -> pivot -> creep` pickup tail
    after pruning, so the route always finishes with an in-place alignment and a
    straight TCP creep to the ball.
  - Added route segment metadata (`TRANSIT`, `PIVOT`, `CREEP`) plus intended
    speed percentages for route rendering and diagnostics.
  - The schematic route now colors segments by intended speed: green transit,
    yellow/orange pivot, and red low-speed creep.
  - Lowered the default TCP near-zone move speed to the configured creep speed
    so the EV3 command path matches the planned terminal profile.

- Retargeted pickup-standoff search heuristics and added progressive fallback.
  - `GridDijkstraHeuristic` can now seed from multiple source nodes.
  - Pickup standoff planning now builds its Dijkstra heuristic from all
    hard-valid standoff poses instead of the ball grid node.
  - If the standard attempt exhausts its expansion budget, the planner retries
    with the ball costmap scaled to 10%, then retries without soft costs using
    `heuristic_weight = 1.0` and a wider flexible heading tolerance.
  - Search logs now print the target ball field coordinate and the number of
    hard-valid standoff candidates.

### Verified

- Ran:

```text
env PYTHONPYCACHEPREFIX=/private/tmp/codex-pycache python3 -m unittest test.test_drive_control
env PYTHONPYCACHEPREFIX=/private/tmp/codex-pycache python3 -m unittest test.test_pathfinding_heuristic
env PYTHONPYCACHEPREFIX=/private/tmp/codex-pycache python3 -m unittest test.test_robot_controller_safety
env PYTHONPYCACHEPREFIX=/private/tmp/codex-pycache python3 -m unittest test.test_robot_server_commands
env PYTHONPYCACHEPREFIX=/private/tmp/codex-pycache python3 -m unittest test.test_topdown_detector_app_shell
```

- Result: all listed test suites passed.
- Added focused regression coverage for Dijkstra costmap penalties, non-target
  ball costmap construction, and removal of the old contact-fallback replanning
  path.
- Added focused regression coverage for protected terminal pivot/creep route
  segment classification and speed metadata.
- Added focused regression coverage for multi-source Dijkstra maps and the
  pickup-standoff soft-cost fallback path.

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
  - Red markers show route-tracking halt / TCP turn locations.
  - Final TCP center-to-center approach segment is drawn using the same speed heatmap palette as the tracked route.
  - Intermediate robot footprint clutter was removed; pickup footprints are shown only at collection poses.
- Added global ball-count reconciliation in `tools/topdown_object_detector.py`.
  - Drive dispatch waits for a stable initial YOLO ball count.
  - Successful TCP pickup moves optimistically increment `balls_collected`.
  - A 15-frame stable count debounce corrects missed/nudged pickups by decrementing `balls_collected`.
  - Debug overlay displays initial count, collected count, and stable visible count.

### Changed

- Rewrote `AGENTS.md` to match the top-down object detection, Hybrid A*, and
  robot-control architecture active at that time.
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
  - near-zone route-tracking stop and TCP turn/move ordering,
  - route heatmap near-zone breaks,
  - pickup standoff geometry,
  - global ball-count reconciliation and debounce behavior.

### Verified

- Ran:

```text
env PYTHONPYCACHEPREFIX=/private/tmp/codex-pycache python3 -m unittest test.test_pathfinding_heuristic test.test_drive_control test.test_topdown_detector_app_shell
```

- Result: `Ran 19 tests ... OK`
