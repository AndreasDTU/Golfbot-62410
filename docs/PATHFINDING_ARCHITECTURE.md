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
selected pickup target, the planner builds a temporary copy of the red-zone
grid and inserts every other currently unvisited ball as an inflated hard
obstacle. The selected target ball is explicitly excluded from that temporary
obstacle layer so its own pickup standoff/final-pickup poses remain reachable.

The inflated ball obstacle radius is:

```text
ball_radius_cm + 0.5 * robot_width_cm + non_target_ball_extra_clearance_cm
```

These defaults are configured in `vision.config.PlannerConfig` and copied into
`pathfinding.models.HybridPlannerConfig`:

- `avoid_non_target_balls_enabled`
- `ball_radius_cm`
- `non_target_ball_extra_clearance_cm`
- `allow_last_resort_orange_contact`

This is a conservative hard-obstacle layer, not a cost-only hint. If a white
ball lies directly between the robot and the selected target, Hybrid A* should
route around the inflated ball region whenever the field layout allows it.
When debugging detection or route geometry, ball avoidance can be disabled
through configuration, but the default behavior is to avoid non-target balls.

## Pickup Offset

The route is expressed in robot body-center coordinates. Ball pickup goals are
generated as paired poses:

- a standoff pose where UDP route tracking stops
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
overhang walls, but the body may not. Hybrid A* targets the set of valid
standoff poses and appends the paired final pickup pose only after the standoff
is reached. This guarantees the final TCP straight-line segment is collinear
with the robot heading and does not require impossible sideways motion.

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

Near pickup targets, the drive loop switches from UDP velocity control to
calibrated TCP position control:

1. Confirm or command `collector_travel_position()` before route following.
2. Stop UDP wheel output with `LR 0 0`.
3. Compute the current heading error to the final pickup pose.
4. Execute blocking TCP `turn(degrees, speedPercent)`.
5. Execute blocking TCP `move(distance, speedPercent)`.
6. Run the collection-only `pickup_assist()` pipe command.
7. Return the collector state to `TRAVEL`.
8. Continue to pickup/replan after the assist motion completes.

The near-zone TCP speed and turn speed are configured through `DriveConfig`.
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

When any of those checks fails, the cache is cleared and the next frame replans.
This keeps the UI responsive while still reacting to target collection, target
motion, and robot drift.

If no ball is reachable, that negative result is cached too. The cache is still
invalidated by ball-set changes or robot drift, which avoids repeating the most
expensive failure case on every video frame.

## Route Visualization

The schematic route is a robot body-center preview. Both the UDP path and the
final TCP near-zone segment are drawn with the same speed heatmap palette.

At each near-zone handoff, a red marker indicates the `LR 0 0` stop and TCP
turn point. The line from the standoff center to the final pickup center uses
the configured TCP move speed for its heatmap color. No route line is drawn to
the ball coordinate, because the robot center never moves there.

Every successful planned pickup pose is drawn as a bold magenta footprint. These
pickup footprints show the base-center offset and intake alignment for each
orange or white target that the route actually connects.

## Orange Ball Priority

`build_greedy_route()` is state-aware:

1. If orange targets exist, they are sorted by distance from the robot.
2. Hybrid A* tries each orange target in nearest-first order with all
   non-target balls inserted as inflated hard obstacles.
3. If no valid orange trajectory exists with ball avoidance enabled, the route
   never changes the selected target to a white ball.
4. Only when `allow_last_resort_orange_contact` is enabled may the planner relax
   the non-target ball obstacle layer for orange as a last resort. The active
   target remains orange, and this mode is labelled `orange forced first` and
   logged so it is visible during contest debugging.
5. After an orange ball is reached, the same nearest-reachable logic continues
   from the final `HybridPose`.

The fallback never fabricates a route. Unreachable targets are skipped, and the
route contains only validated Hybrid A* segments.
