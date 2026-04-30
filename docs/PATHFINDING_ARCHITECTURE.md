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
- forward left/right arcs
- in-place left/right turns

The returned route is a trajectory of `HybridPose` values, so downstream drive
logic can follow both position and heading instead of only a polyline.

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

The schematic draws the route polyline plus sparse heading markers. Each marker
shows a base dot and an arrow toward the intake mouth, making the `theta_rad`
component of the Hybrid A* trajectory visible without overcrowding the view.

## Orange Ball Priority

`build_greedy_route()` is state-aware:

1. If an orange target exists, Hybrid A* tries to route to it first.
2. If no valid orange trajectory exists in the current map, the orange target is
   treated as unreachable for that planning pass.
3. Routing then falls back to nearest-reachable greedy collection for the
   remaining balls.
4. After the orange ball is reached, the same nearest-reachable logic continues
   from the final `HybridPose`.

The fallback never fabricates a route. Unreachable targets are skipped, and the
route contains only validated Hybrid A* segments.
