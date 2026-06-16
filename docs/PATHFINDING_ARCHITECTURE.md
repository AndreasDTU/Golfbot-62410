# Pathfinding Architecture

Status: ACTIVE KEPT LAYER

Path is one of the grey layers in the movement-layer architecture. It is kept
code. The previous top-down detector app that called it has been deleted during
the movement rework, so this document describes the Path layer itself, not a
currently available autonomous app.

## Current Locations

```text
path/pathfinding/models.py
path/pathfinding/planner.py
path/pathfinding/plancreation.py
path/tools/pathfinding_sandbox.py
```

Imports should use:

```text
path.pathfinding.*
```

## Planner

Routing uses Hybrid A* over:

```text
x_cm, y_cm, theta_rad
```

`x_cm` and `y_cm` are continuous field coordinates in centimeters with the same
bottom-left origin used by robot pose estimation. `theta_rad` is the robot
heading, where zero points along positive X.

The closed set discretizes states to 1 cm cells and heading bins. Neighbor
expansion uses deterministic short motion primitives including straight moves
and in-place rotations.

The returned route is a trajectory of `HybridPose` values, so downstream layers
can follow both position and heading.

## Collision Model

Red zones are represented in a 1 cm occupancy grid. For each candidate pose,
the collision checker evaluates an oriented multi-circle approximation of the
main robot body. The pickup tube is intentionally excluded from some boundary
and red-zone checks so it can overhang walls during valid pickups while the main
body remains collision-free.

The planner also supports soft costs around non-target balls. The selected
target ball is excluded from its own avoidance layer so pickup poses remain
reachable.

## Pickup Offset

Routes are robot body-center trajectories. Pickup goals are generated as paired
poses:

- a standoff pose where route tracking should stop,
- a final pickup pose where the tube center reaches the ball.

The final pickup pose is computed from the ball coordinate, robot heading, tube
forward offset, and tube lateral offset. The standoff pose is translated
backward from that final pickup pose along the same heading.

Both poses must be valid for the main robot body. The terminal pickup maneuver
is represented as a protected tail:

```text
standoff pose -> in-place pivot pose -> straight TCP creep final pose
```

Do not replace this with raw ball-coordinate routing.

## Route Model

Core route models live in `path/pathfinding/models.py`:

- `HybridPose`
- `PlannedBallTarget`
- `RoutePlan`
- `RouteSegmentType`
- `RouteTrackingError`
- `HybridPlannerConfig`

`RouteSegmentType` can describe:

- `TRANSIT`
- `PIVOT`
- `CREEP`

Those segment types are route metadata for future Guidance/Control layers; they
do not mean the Brain/Guidance/Control stack is currently rebuilt.

## Orange Ball Priority

The greedy route planner preserves orange-first target selection. If orange
targets exist, it tries reachable orange routes before white routes. If no valid
orange trajectory exists, the selected target should not silently change to a
white ball in the same orange-first pass.

## Current Integration Status

Path is import-clean in the new layer layout and covered by the remaining
pathfinding tests. It is not currently wired to a complete autonomous Brain
because the old app shell was removed.

Run current tests with:

```bash
env PYTHONPYCACHEPREFIX=/private/tmp/codex-pycache /Users/alex/miniforge3/bin/python3 -m unittest discover -s test -p 'test_*.py'
```
