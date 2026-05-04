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
Touching is allowed. A base circle rejects a pose only when its distance to the
nearest red cell is strictly less than its radius, so grazing walls or the
center cross does not create an artificial safety-margin failure.

This keeps clearance deterministic and orientation-aware without adding runtime
geometry dependencies beyond OpenCV and NumPy.

## Pickup Offset

The route goal is the intake tip, not the robot origin. Once Hybrid A* finds a
feasible approach heading near a ball, the final waypoint is snapped to the
exact base-center pose:

```text
base_target_x = ball_x - cos(theta) * intake_length
base_target_y = ball_y - sin(theta) * intake_length
```

`intake_length` is the tuned `tube_forward_cm`, the physical pivot-to-pickup
distance. If a lateral tube offset is tuned, it is also subtracted so the
visualized pickup point still lands on the ball.

## Route Cache

Hybrid A* is not recomputed every frame. The detector keeps the active route and
reuses it while:

- the active target ball still exists
- the target has not moved significantly
- the intake has not reached the target
- the robot remains close to the cached trajectory

## Integrated Control

`tools/topdown_object_detector.py` is now the master controller. The deprecated
external `autonomous_navigator.py` flow is not required for closed-loop driving.
After the required vision pipeline and Hybrid A* route update, the detector:

- estimates the live robot pose from calibrated ArUco markers
- projects the robot origin onto the closest cached Hybrid A* route segment
- computes cross-track error (XTE) as the shortest segment distance
- computes heading error against that segment heading
- converts those errors into bounded left/right differential-drive speeds
- dispatches the wheel-speed command directly to the robot microcontroller

Dispatch uses best-effort non-blocking UDP so the OpenCV frame loop never waits
for robot acknowledgements. The default command payload is:

```text
LR <left_speed_pct> <right_speed_pct>
```

The robot endpoint is configured in `topdown_object_detector.py` with
`ROBOT_IP`, `ROBOT_UDP_PORT`, and `ROBOT_COMMAND_FORMAT`. The `--drive` flag is
the only runtime switch for hardware dispatch. Without `--drive`, the controller
still computes and visualizes XTE and motor commands, but no hardware packets
are sent.

If XTE exceeds `MAX_CROSS_TRACK_ERROR_CM` (8 cm by default), the detector sends
a zero-speed STOP command, invalidates the cached route, and immediately runs a
fresh Hybrid A* search from the deviated robot pose. Motor output resumes only
after a valid route is cached again.

When any of those checks fails, the cache is cleared and the next frame replans.
This keeps the UI responsive while still reacting to target collection, target
motion, and robot drift.

If no ball is reachable, that negative result is cached too. The cache is still
invalidated by ball-set changes or robot drift, which avoids repeating the most
expensive failure case on every video frame.

## Heading Visualization

The schematic draws the yellow route polyline and sparse cyan heading arrows by
default. Intermediate footprint snapshots are controlled by
`NUM_INTERMEDIATE_SNAPSHOTS`, which defaults to `0` to keep the UI uncluttered.

Every successful planned pickup pose is drawn as a bold magenta footprint, not
just the final endpoint of the greedy route. These pickup footprints show the
base-center offset and intake alignment for each orange or white target that
the route actually connects.

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
