# Fit-Based Pathing

## Core Idea

Generate discrete pickup points on a ring around each ball, then classify
them by proximity to the nearest obstacle. The classification determines
how the robot may approach and leave each point, guaranteeing that an
in-place (tank) turn never swings the robot body into an obstacle.

## Hard Constraints

1. **Orange first** -- the orange ball must always be visited first in
   the route (competition bonus points).
2. **Start at robot** -- the route always begins from the robot's current
   position (`start_pose`).

## Robot Geometry Reference

All measurements from origin (wheel axle center).
Ground truth lives in `data/robot_calibration.json`.

```
             O          <- tube tip (4.5 cm diameter pipe)
             |
             |  tube_forward_cm = 13.1 (to pipe center)
             |
    +--------+--------+  <- front_cm = 3.8
    |        |        |
    |       (O)       |  <- origin
    |                 |
    |                 |
    |                 |  rear_cm = 15.1
    +-----------------+
    <-   width_cm    ->
          19.5
```

The pickup tube is **not retractable** -- it protrudes forward at all times
and must be included in collision geometry.

### Tank Turn Swept Radii

| Point              | Distance from origin | Formula                                        |
|--------------------|---------------------:|------------------------------------------------|
| Rear corner (max)  |             18.0 cm  | `sqrt(15.1^2 + 9.75^2)`                       |
| Tube tip edge      |             13.3 cm  | `sqrt(13.1^2 + 2.25^2)`                       |
| Front body corner  |             10.5 cm  | `sqrt(3.8^2 + 9.75^2)`                        |

**Tank turn radius = 18.0 cm** (dominated by rear corners).

## Pickup Point Categories

Each pickup point on the ring (radius = `tube_forward_cm` = 13.1 cm from
ball center) is classified by its distance `d` to the nearest obstacle.

### Safe -- `d > safe_radius`

- `safe_radius = max(tank_turn_radius, tube_sweep_radius)`.
- Full 360 degree rotation is collision-free.
- Robot can approach from any direction.
- Tank turn at the pickup point to align with ball, pick up, tank turn to
  face next target, drive away.

### In-Between -- `constrained_radius < d <= safe_radius`

- `constrained_radius = min(tank_turn_radius, tube_sweep_radius)`.
- One swept component clears, the other does not.
- **Policy: use an intermediate node** (same as constrained).
- The safe angular range could be computed per-point, but the band is
  narrow -- the complexity is not worth it.

### Constrained -- `d <= constrained_radius`

- Neither component clears a rotation.
- Robot **must** arrive already on the correct heading.
- Requires an intermediate node placed along the approach heading in the
  safe zone (`d > safe_radius`).

## Intermediate Node Mechanics

For every non-safe pickup point:

1. **Placement:** Walk backward from the pickup point along its heading
   until distance to nearest obstacle exceeds `safe_radius`. That
   position is the intermediate node.
2. **Approach:** The route planner connects intermediate nodes (all in safe
   zones, so standard pathfinding applies).
3. **Execution sequence:**
   - Drive to intermediate node.
   - Tank turn to face the ball.
   - Drive straight to pickup point.
   - Pick up ball.
   - Reverse straight back to intermediate node.
4. **Fallback:** If no intermediate node can be placed within ~30 cm, try
   the next candidate heading. If no heading works, the ball is
   unreachable.

## Design Advantages

- **Safety by construction.** Tank turns only happen where the full swept
  circle is obstacle-free. No reliance on the planner to avoid collisions
  during rotation.
- **Uniform execution model.** Every pickup is one of two patterns --
  direct (safe) or via intermediate node. The route interpreter needs no
  special cases.
- **Replaces existing complexity.** Wall-pickup perpendicular heuristics,
  corner-ball diagonal approaches, and flexible standoff logic can all be
  deleted in favor of this single system.

## Known Considerations

- **Back-and-forth cost.** Two consecutive constrained pickups require
  driving to intermediate -> pickup -> reverse -> drive to next intermediate.
  Acceptable but the route optimizer should account for the extra distance.
- **Intermediate node conflicts.** Nearby constrained balls may have
  intermediate nodes at similar positions. These must remain separate nodes
  in the route graph even if they overlap spatially.
- **Multi-obstacle pinch.** A ball near the cross or in a corner may have
  very few valid headings, all requiring intermediate nodes. As long as one
  intermediate is reachable, the ball is collectible. If none are, the ball
  is correctly classified as unreachable.
- **Tube width discrepancy.** `PlannerConfig.tube_width_cm` is 6.0 cm but
  the physical pipe diameter is 4.5 cm. For collision sweep calculations
  the physical 4.5 cm must be used. The 6.0 cm value may be a pickup
  planning tolerance -- needs clarification.

## Path Layer Architecture

Three sub-layers, one file each. Each layer has a clean input/output
contract. Data flows strictly downward: 1 -> 2 -> 3.

The previous design had four sub-layers (Route Builder + Pathfinding).
These were merged into a single Route Compiler -- see
`docs/refactor/movement_layer_architecture.md` for the rationale.

```
+-----------------------------------------------------------+
|  Layer 1: Pickup Geometry          pickup_geometry.py      |  COMPLETE
|  balls + obstacles + robot geometry                        |
|  -> classified pickup candidates + distance field          |
+-----------------------------------------------------------+
|  Layer 2: Route Strategy             route_strategy.py     |  COMPLETE
|  candidates + distance field + robot pose                  |
|  -> ordered candidate selection + intermediate nodes       |
+-----------------------------------------------------------+
|  Layer 3: Route Compiler                   planner.py      |  DESIGNED
|  ordered stops + obstacle map                              |
|  -> collision-free annotated waypoint graph                |
+-----------------------------------------------------------+

Visualizer:  pickup_visualizer.py  (isolation testing)         COMPLETE
```

### Layer 1 -- Pickup Geometry (`pickup_geometry.py`) -- COMPLETE

**Input:** ball positions, obstacle map, robot geometry.

**Responsibility:**
- Compute obstacle distance field (distance transform).
- Merge wall borders into obstacle grid (occupancy grid only has red
  zones -- walls must be added before the distance transform).
- Generate pickup ring (radius = `hypot(tube_forward, tube_right)`)
  per ball, with mouth-radius tolerance offsets.
- Sample discrete headings (72 = 5 degree increments) around each ring.
- Filter out candidates where robot origin is too close to obstacles
  (`dist < half_width`).
- Classify each surviving candidate as safe / in-between / constrained
  using `safe_radius = max(tank_turn, tube_sweep)` and
  `constrained_radius = min(tank_turn, tube_sweep)`.
- Track rejected headings as `InvalidHeading` for visualization.

**Output:** `PickupGeometryResult`:
- `balls: tuple[BallCandidates, ...]` -- per-ball candidates + invalids
- `distance_field: np.ndarray` -- full distance transform (cm/px)
- `tank_turn_radius_cm`, `tube_sweep_radius_cm`
- `safe_radius_cm`, `constrained_radius_cm`
- `ring_radius_cm`

Each `PickupCandidate` contains:
- `x_cm, y_cm` -- robot origin position
- `theta_rad` -- heading (faces the ball)
- `category` -- safe / in_between / constrained
- `obstacle_distance_cm` -- distance to nearest obstacle
- `ball_index` -- which ball this candidate picks up

**Does NOT:** choose between candidates, place intermediate nodes, or
decide visit order.

### Layer 2 -- Route Strategy (`route_strategy.py`) -- COMPLETE

**Input:** `PickupGeometryResult` from layer 1, robot start pose, unload
position.

**Responsibility:**
- Select one candidate per ball (prefer SAFE -> IN_BETWEEN -> CONSTRAINED,
  tiebreak by descending `obstacle_distance_cm`).
- For non-safe candidates: place intermediate nodes by walking backward
  along the approach heading until `d > safe_radius_cm` (max 30 cm).
  If no intermediate found, try the next candidate heading.
- Fallback: if no candidate has a valid intermediate, use the best
  candidate without an intermediate (last resort).
- **Orange-first**: always visit the orange ball first (hard constraint).
- Nearest-neighbor greedy ordering from `start_pose` for remaining balls.

**Output:** `RouteStrategyResult`:
- `stops: tuple[RouteStop, ...]` -- ordered visit sequence
- `unreachable_balls: tuple[int, ...]`
- `unload_position: tuple[float, float] | None`

Each `RouteStop` contains:
- `candidate: PickupCandidate` -- chosen approach
- `intermediate_node: HybridPose | None` -- None if SAFE
- `ball_index: int`

**Does NOT:** build waypoint sequences or plan paths.

### Layer 3 -- Route Compiler (`planner.py`)

Merges the old Route Builder and Pathfinding layers into one. Full
specification in `docs/refactor/movement_layer_architecture.md`.

**Input:**
- `RouteStrategyResult` from layer 2
- `distance_field: np.ndarray` from layer 1 (cm per pixel)
- `start_pose: HybridPose`
- `half_width_cm: float` (~9.75 cm)

**Key design rule:** the path is intentionally dumb. It describes
*where* to go and *what* to do there. It never prescribes *how* to
move. Brain decides intent. Guidance decides geometry.

**Responsibility:**

1. Walk ordered stops -> determine positions to visit per stop:
   - Constrained/in-between: intermediate -> pickup -> intermediate
   - Safe: pickup only

2. Between each consecutive pair of positions, check the straight line
   using the distance field. If `min(d) >= half_width` along the line,
   connect directly. Otherwise, A* on the distance field
   (`d >= half_width` = passable). Simplify A* paths with
   Douglas-Peucker, re-verifying each simplified segment.

3. Annotate each waypoint: NAVIGATE / PICKUP / UNLOAD.

4. Append unload waypoints at the end.

5. Assign headings: A* waypoints get `atan2(dy, dx)` toward the next
   waypoint. Destination waypoints keep their original heading.

**Output:** `RoutePlan` -- flat, ordered list of `RouteWaypoint`s.
Each waypoint carries a `WaypointKind` (NAVIGATE / PICKUP / UNLOAD)
and nothing else. **No movement semantics.**

**Collision avoidance:** uses the distance field from layer 1 as the
single source of truth. No separate grid dilation. A position is
traversable if `distance_field[row, col] >= half_width_cm`.

**v1 limitations:**
- Uncollected balls are not treated as obstacles (deferred to v2).
- Robot body approximated as circle of radius `half_width` (safe but
  conservative).

**Does NOT:** prescribe movement mode (forward/reverse/turn), encode
action sequences, or carry movement instructions of any kind.

### Visualizer (`pickup_visualizer.py`) -- COMPLETE

Standalone tool for testing layers 1-2 without the robot or camera.

**Displays:**
- Field boundaries and obstacles (cross, walls).
- Obstacle distance field as a semi-transparent heatmap.
- Balls with pickup rings (cyan if reachable, red if not).
- Pickup candidates color-coded: green=safe, yellow=in-between,
  red=constrained, black=invalid headings.
- Intermediate nodes (orange squares) with dashed lines to pickup points.
- Route polyline with arrowheads and numbered stop markers.
- Unload marker (magenta triangle).
- Robot start marker (orange arrow, positioned opposite delivery node).
- Compact single-line legend.

**Test mode:** `--seed=N` with edge-biased ball placement (balls near
field walls, simulating competition conditions). Press `r` to randomize.

## Open Topics

- Layer 3 Route Compiler implementation -- design finalized, code in
  `planner.py` needs rewriting. `route_builder.py` will be deleted
  (its responsibilities are absorbed into the Route Compiler).
- `RoutePlan` output type needs updating: replace `id()` matching with
  explicit `WaypointKind` annotations (see architecture doc).
- `route_interpreter.py` needs updating to split on `WaypointKind`
  instead of `id()` identity matching.
- Unload trigger logic (capacity-based mid-route unload).
- Body clearance check in layer 1 uses simplified circle (half_width)
  rather than oriented rectangle -- acceptable for ring sampling but
  worth noting.
- v2: add uncollected balls to the distance field before collision
  checking so the route avoids them.
