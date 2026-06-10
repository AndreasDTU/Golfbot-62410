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

Detected balls are also considered during target-specific planning. For each
selected pickup target, the planner keeps the red-zone grid as the hard
`uint8` obstacle layer and builds a parallel `float32` ball costmap for every
other currently unvisited ball. The selected target ball is explicitly excluded
from that soft-cost layer so its own pickup standoff/final-pickup poses remain
reachable.

The core ball-cost radius is:

```text
ball_radius_cm + 0.5 * robot_width_cm + non_target_ball_extra_clearance_cm
```

The planner draws three concentric cost bands with OpenCV:

```text
core radius:        ball_core_cost
core radius + 5cm:  ball_close_cost
core radius + 10cm: ball_warning_cost
```

The default costs are `1000.0`, `200.0`, and `50.0`. These defaults are
configured in `vision.config.PlannerConfig` and copied into
`pathfinding.models.HybridPlannerConfig`:

- `avoid_non_target_balls_enabled`
- `ball_radius_cm`
- `non_target_ball_extra_clearance_cm`
- `ball_core_cost`
- `ball_close_cost`
- `ball_warning_cost`
- `ball_close_clearance_cm`
- `ball_warning_clearance_cm`

This is a soft traversal-cost layer, not an impassable obstacle layer. Hybrid A*
adds the costmap value at each candidate robot-center pose into `g(n)`, and the
2D Dijkstra heuristic adds the same cost layer during node expansion. During
pickup-standoff planning, that Dijkstra heuristic is seeded from all
hard-valid standoff grid nodes instead of the ball grid node, so the heuristic
pulls the body center toward a reachable TCP handoff pose rather than into the
ball itself. This makes routes through non-target balls very expensive while
still allowing them if a ball corridor is the only route to the selected target.
Red zones and field boundaries remain hard constraints.

## Search Cost Tuning

Hybrid A* uses weighted A* for route planning. The priority queue score is:

```text
f(n) = g(n) + heuristic_weight * h(n)
```

`heuristic_weight` defaults to `1.5`, which intentionally favors faster,
greedier route discovery over perfectly minimal path cost. This is useful in the
live contest loop because route latency matters more than tiny path optimality
differences.

Motion primitive costs are also tuned to reduce wiggle:

- forward move: `step_cm`
- reverse move: `step_cm * reverse_cost_multiplier`
- gear shift: add `gear_shift_penalty` when switching directly between forward
  and reverse
- steering change: add `steering_change_penalty` when entering a different
  in-place turn direction
- in-place turn: `in_place_rotation_cost + abs(delta_theta) * 0.25`

Current defaults are `reverse_cost_multiplier = 2.5`,
`gear_shift_penalty = 50.0`, `steering_change_penalty = 3.0`, and
`in_place_rotation_cost = 2.0`. The previous gear and steering direction are
tracked internally for the best-known arrival at each `(x, y, theta)` state;
they are not part of the public route model.

## Pickup Offset

The route is expressed in robot body-center coordinates. Ball pickup goals are
generated as paired poses:

- a standoff pose where TCP route tracking stops
- a final pickup pose where the TCP encoder move ends and the tube center is on
  the ball

The final pickup pose is computed from each candidate approach heading:

```text
base_target_x = ball_x - cos(theta) * intake_length
base_target_y = ball_y - sin(theta) * intake_length
```

`intake_length` is the tuned `tube_forward_cm`, the physical pivot-to-pickup
distance. If a lateral tube offset is tuned, it is also subtracted so the
visualized pickup point still lands on the ball.

The standoff pose is then computed from the final pickup pose, not radially from
the ball:

```text
standoff_x = base_target_x - cos(theta) * near_zone_cm
standoff_y = base_target_y - sin(theta) * near_zone_cm
standoff_theta = theta
```

Both poses must be collision-free for the main robot body. The intake tube may
overhang walls, but the body may not. Hybrid A* can still target the fixed
standoff/final-pickup pose pairs, but the near-ball goal condition is flexible:
if the current pose has the tube within `flexible_standoff_max_cm` of the ball,
the body heading points at the ball within `flexible_standoff_heading_tolerance_rad`,
and the straight TCP segment to the final pickup pose is hard-obstacle clear,
the planner accepts that pose as the handoff point. This lets the robot stop,
pivot, and start the TCP sneak immediately when it is already close instead of
backing up to exactly 15 cm.

Before a route segment is returned, Hybrid A* applies a greedy pruning pass. A
middle node is removed when the anchor node has a sampled collision-free straight
segment to a later node and that shortcut does not enter a worse soft-cost band
than the original subpath. This trims grid-search jaggedness while preserving
hard red-zone/wall safety.

The terminal pickup maneuver is appended only after that pruning pass. This
protected tail is always represented as:

```text
standoff pose -> in-place pivot pose -> straight TCP creep final pose
```

The pivot pose has the same robot-center `(x, y)` as the standoff but faces the
target ball directly. The final pose is then the straight-line creep endpoint
whose tube center lands on the ball. This keeps smoothing from removing the
controlled TCP creep sequence.

Pickup-standoff search uses progressive fallback when the standard soft-cost
attempt exhausts its expansion budget:

1. Standard weighted A* with normal soft ball costs.
2. Relaxed soft costs, with the ball costmap scaled to 10% of its original
   values.
3. Desperation mode, with no soft ball costs, `heuristic_weight = 1.0`, and the
   flexible handoff heading tolerance widened to 30 degrees.

These fallbacks never relax hard red-zone, wall, or robot-footprint validity.
If no hard-valid standoff/final pickup pair exists, the target is still rejected
before search begins.

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

Dispatch uses the EV3 TCP command server for the entire `--drive` scenario,
including continuous route-following wheel speeds and calibrated pickup moves.
The default route-following command payload is:

```text
LR <left_speed_pct> <right_speed_pct>
```

The robot endpoint is configured through `DriveConfig.robot_ip`,
`DriveConfig.robot_tcp_port`, and `DriveConfig.robot_command_format`. The
`--drive` flag is the only runtime switch for hardware dispatch. Without
`--drive`, the controller still computes and visualizes XTE and motor commands,
but no hardware commands are sent.

If XTE exceeds `MAX_CROSS_TRACK_ERROR_CM` (8 cm by default), the detector sends
a zero-speed STOP command, invalidates the cached route, and immediately runs a
fresh Hybrid A* search from the deviated robot pose. Motor output resumes only
after a valid route is cached again.

Near pickup targets, the drive loop switches from TCP wheel-speed tracking to
calibrated TCP position control on the same command server:

1. Stop TCP wheel output with `LR 0 0`.
2. Compute the current heading error to the final pickup pose.
3. Execute blocking TCP `turn(degrees, speedPercent)`.
4. Enter a visual-servo loop that consumes one refreshed pose/ball observation
   per frame.
5. While alignment is still improving, compute the lateral ball error in the
   robot frame and execute proportional TCP micro-turns. Turn angle and speed
   shrink as the measured error approaches zero.
6. Stop aligning only when the lateral error reaches the configured camera
   noise floor, improvement stalls for the configured number of frames, or the
   bounded iteration limit is reached.
7. Force a zero-speed TCP stop, wait for the configured settling interval, and
   verify the lateral error on a fresh camera frame.
8. Execute blocking TCP `move(distance, speedPercent)` only if post-settle
   verification still passes.
9. Run the collection-only `pickup_assist()` pipe command.
10. Continue along the cached global route unless visible remaining balls no
   longer match the original plan.

The near-zone TCP turn speed is configured through `DriveConfig`; the TCP move
speed defaults to the same low creep speed used in planner segment metadata.
The collection actuator contract is documented in
`docs/COLLECTION_MECHANISM.md`: autonomous ball collection, including
orange-first collection, may use only the small pickup-assist motion. The full
`unload_full_cycle()` pipe motion is reserved for unloading at the goal and must
not be called from the collection route logic.

Before `--drive` dispatch is allowed, the detector also waits for a stable
initial YOLO ball count. After each TCP pickup move it optimistically increments
`balls_collected`, then reconciles against a debounced rolling visible-ball
count. If stable visible balls exceed the expected visible count, the pickup is
treated as missed and `balls_collected` is decremented so the ball can be routed
again.

Pickup completion no longer clears the route unconditionally. The detector
keeps following the cached global route when the current visible balls remain a
stationary subset of the balls used for the plan. It clears/replans only when a
remaining ball crosses the configured target-move bucket, a new/previously
hidden ball appears, robot drift invalidates the route, or route metadata no
longer matches the active geometry.

If `balls_collected > 0` and the current camera frame contains zero visible
balls, the route planner may receive an empty target list as an explicit request
to route directly from the current robot pose to the small-goal unload pose.

Before the autonomous unload pipe sequence is allowed to run, the drive loop now
performs reverse visual servoing against the fixed small-goal center
`(0.0, field.height_cm / 2)`. The controller computes the live rear unload tip
from `rear_cm + unload_extension_cm`, turns so the robot rear faces the goal,
and issues small TCP `move`/`back` corrections until the rear tip is within the
configured visual-servo noise floor. The final drop is accepted only after a
forced zero-speed stop, the configured settling interval, and a fresh-frame
verification that the rear tip is still on the goal and the robot heading is
within the allowed left-wall unload arc.

If no ball is reachable, that negative result is cached too. The cache is still
invalidated by ball-set changes or robot drift, which avoids repeating the most
expensive failure case on every video frame.

## Route Visualization

The schematic route is a robot body-center preview. Route edges carry semantic
segment metadata:

- `TRANSIT`: normal route-following movement at the configured transit speed
- `PIVOT`: in-place rotation at the configured pivot speed
- `CREEP`: final straight TCP pickup movement at the configured creep speed

The OpenCV schematic draws segments from that speed profile: low speed is red,
medium/pivot speed is yellow-orange, and high transit speed is green. Zero-length
pivot edges are shown as colored stop markers so the intended in-place alignment
is still visible.

At each terminal handoff, the pivot marker indicates the `LR 0 0` stop and TCP
turn point. The line from the pivot center to the final pickup center uses the
configured creep speed for its heatmap color. No route line is drawn to the ball
coordinate, because the robot center never moves there.

Every successful planned pickup pose is drawn as a bold magenta footprint. These
pickup footprints show the base-center offset and intake alignment for each
orange or white target that the route actually connects.

## Orange Ball Priority

`build_greedy_route()` is state-aware:

1. If orange targets exist, they are sorted by distance from the robot.
2. Hybrid A* tries each orange target in nearest-first order with all
   non-target balls represented in the soft ball costmap.
3. If no valid orange trajectory exists with ball avoidance enabled, the route
   never changes the selected target to a white ball.
4. After an orange ball is reached, the same nearest-reachable logic continues
   from the final `HybridPose`.

For normal white-target routing, each candidate is tried once with the soft
non-target ball costmap. There is no second contact-fallback route pass; if the
only feasible trajectory crosses a non-target ball cost band, the same Hybrid A*
search can still choose it by paying the high traversal cost. Unreachable
targets are skipped, and the route contains only validated Hybrid A* segments
against walls and red zones.
