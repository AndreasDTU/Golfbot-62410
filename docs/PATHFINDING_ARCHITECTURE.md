# Pathfinding Architecture

Status: ACTIVE

## Scope

`tools/topdown_object_detector.py` owns the interactive route preview used by
the top-down detector. The vision pipeline still produces the same red-zone and
ball detections, then maps them into field centimeters. Routing begins after
grid mapping and must not reorder or mutate the vision pipeline.

## Planner

Routing now uses Hybrid A* instead of a 2D point-grid A* planner.

Each search state is:

```text
x_cm, y_cm, theta_rad
```

`x_cm` and `y_cm` are continuous field coordinates in centimeters with the same
bottom-left origin used by robot pose estimation. `theta_rad` is the robot
heading, where zero points along positive X.

The closed set discretizes states to 1 cm cells and a fixed number of heading
bins. Neighbor expansion uses deterministic short motion primitives:

- forward straight
- pure in-place left/right rotations

The returned route is a trajectory of `HybridPose` values, so downstream drive
logic can follow both position and heading instead of only a polyline.

The robot is differential drive. The planner therefore uses rotate-then-drive
sequences instead of Ackermann-style steering arcs. In-place rotation has a low
cost so the planner can reorient in tight spaces without inventing sweeping
turns the drivetrain does not need.

Search has a hard expansion cap. If the cap is hit, the target is treated as
unreachable for the current planning pass and routing continues with the next
candidate.

## Collision Model

Red zones are stored in a raw 1 cm occupancy grid. They are no longer dilated by
a circular robot radius.

For every Hybrid A* candidate pose, the collision checker computes a fast
oriented multi-circle approximation of the robot footprint:

- a wide rectangular base/wheelbase
- a narrow forward intake tube

The red occupancy grid is converted once per search into a distance transform.
Each base circle then performs a constant-time lookup against that map. Only the
base circles are safety-critical for red-zone clearance. This is intentional:
the intake is allowed to approach red borders so the robot can reach edge balls,
but the wheelbase must stay inside the field and outside raw red occupancy.

This keeps clearance deterministic and orientation-aware without adding runtime
geometry dependencies beyond OpenCV and NumPy.

## Route Cache

Hybrid A* is not recomputed every frame. The detector keeps the active route and
reuses it while:

- the active target ball still exists
- the target has not moved significantly
- the intake has not reached the target
- the robot remains close to the cached trajectory

When any of those checks fails, the cache is cleared and the next frame replans.
This keeps the UI responsive while still reacting to target collection, target
motion, and robot drift.

If no ball is reachable, that negative result is cached too. The cache is still
invalidated by ball-set changes or robot drift, which avoids repeating the most
expensive failure case on every video frame.

## Heading Visualization

The schematic draws the route polyline plus footprint snapshots. Light snapshots
show the base and intake at regular waypoints, and the final pickup pose is
highlighted strongly so intake alignment and base clearance can be checked
against the target ball and red zones.

## Orange Ball Priority

`build_greedy_route()` is state-aware:

1. If orange targets exist, they are sorted by distance from the robot.
2. Hybrid A* tries each orange target in nearest-first order.
3. If no valid orange trajectory exists in the current map, those orange targets
   are treated as unreachable for that planning pass.
4. Routing then falls back to nearest-reachable greedy collection for the
   remaining balls.
5. After an orange ball is reached, the same nearest-reachable logic continues
   from the final `HybridPose`.

The fallback never fabricates a route. Unreachable targets are skipped, and the
route contains only validated Hybrid A* segments.
