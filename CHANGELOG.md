# Changelog

All notable larger additions and behavioral changes to this repository should be recorded here.

## 2026-05-21

### Added

- Added step-by-step drive mode to `tools/topdown_object_detector.py`.
  - New `--step` flag works with `--drive` and starts robot motion in a paused operator-waiting state.
  - Pressing `n` releases one autonomous target run while vision, smoothing, occupancy-grid updates, and async route planning continue every frame.
  - The drive loop automatically pauses again after the existing pickup completion/replan transition.
  - Main debug output now shows a high-visibility paused prompt during step-mode waits.

### Verified

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
