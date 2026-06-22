# Movement Rework · Architecture

Proposed Layer Structure — **v0.2** · 19 Jun 2026.

Supersedes the v0.1 PDF. Changes from v0.1: Path layer internals
refined after fit-based pathing implementation exposed a
responsibility leak between Path and Brain/Guidance.

## Guiding principle

Each layer is named for the single thing it **owns**, so there is
never a question of where a responsibility lives.

The Path layer owns the **route graph** — where to go and what to do
there. It does not own **how to move** — that is Brain (intent) and
Guidance (geometry/commands).

```
┌──────────────────────────────────────────────────────────┐
│  Perception         detect balls + robot → field coords  │  kept
├────────────┬─────────────────────────────────────────────┤
│ Localization│  Path                                      │
│ pose/heading│  route graph + headings                    │  kept
│ freshness   │                                            │
├─────────────┴────────────────────────────────────────────┤
│  Brain (FSM)        intent · route cursor · arbitration  │  new
├──────────────────────────────────────────────────────────┤
│  Guidance           intent + live pose → geometry         │  new
├──────────────────────────────────────────────────────────┤
│  Control            execution · safety gate · backend     │  new
├ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┤
│  EV3 robot                                               │  hardware
└──────────────────────────────────────────────────────────┘

Downward: data/commands.  Upward: status (running / done / failed).
```

---

## The layers

### Perception (kept)

Detects balls and the robot and outputs their positions in field
centimetres. In: camera. Out: detections.

### Localization (new · thin)

Turns detections into a clean robot pose — heading estimation,
smoothing, and a freshness / validity flag so everything downstream
knows whether to trust the pose this frame. In: detections. Out:
RobotPose + valid.

### Path (kept)

Converts the coordinate field into a **route graph with headings**,
plus target selection and pickup / unload structures. In: coordinates.
Out: route graph.

**The path is intentionally dumb.** It describes *where* the robot
should go and *what* it should do at each position. It never
prescribes *how* to move — no "drive straight," no "reverse," no
"tank turn." Those are movement decisions that belong to
Brain/Guidance, which have the live pose and can react in real time.

See [Path layer internals](#path-layer-internals) below.

### Brain — FSM (new)

Decides **intent** — follow the route, turn to the next waypoint, pick
up, unload, recover. Owns the route-progress cursor and arbitration
between modes. Emits intent, not motor commands. In: pose + route.
Out: intent.

### Guidance (new)

Computes the **geometry**: given an intent and the live pose, produces
turn / drive / adjust commands. In: intent + pose. Out: geometric
commands.

### Control (new)

Executes commands on the EV3 — speed profiles, "close enough"
thresholds, motor translation. Hosts the safety / validity gate as its
final stage and a real-vs-simulated backend behind one command API. In:
turn / drive / adjust. Out: motor commands, plus status upward.

---

## Contracts that cross the boundaries

- **Status is a loop, not a pipe.** Every layer below the Brain
  returns running / done / failed upward; the Brain cannot choose the
  next intent without knowing the current one finished.
- **Perception fans out.** It feeds Localization and Path in parallel,
  and the Brain pulls from both — they are not in series.
- **Units and frames are a written contract.** cm, radians, heading
  zero = +X, positive = CCW. Pinned once.
- **Safety / validity gate.** Lives as Control's final stage — stale
  pose, off-field, or an un-executable command stops here.
- **Simulation seam.** Control hides a real-vs-simulated backend
  behind one command API.

---

## The open hinge (resolved)

Non-blocking ticking. Guidance runs every frame, reads the live pose,
and calls one Control command per tick. The six-layer split holds as
drawn and the Brain stays live. *(Decision recorded 16 Jun 2026.)*

---

## Path layer internals

The Path layer has its own internal sub-layers. Data flows strictly
downward: 1 → 2 → 3.

```
┌─────────────────────────────────────────────────────────┐
│  Layer 1: Pickup Geometry          pickup_geometry.py   │  ✅ COMPLETE
│  balls + obstacles + robot geometry                     │
│  → classified pickup candidates + distance field        │
├─────────────────────────────────────────────────────────┤
│  Layer 2: Route Strategy             route_strategy.py  │  ✅ COMPLETE
│  candidates + distance field + robot pose               │
│  → ordered candidate selection + intermediate nodes     │
├─────────────────────────────────────────────────────────┤
│  Layer 3: Route Compiler                   planner.py   │  📐 DESIGNED
│  ordered stops + obstacle map                           │
│  → collision-free annotated waypoint graph              │
└─────────────────────────────────────────────────────────┘
```

### Why three layers, not four

The previous design had four sub-layers: Pickup Geometry → Route
Strategy → Route Builder → Pathfinding. The Route Builder expanded
stops into action sequences (DRIVE_TO, DRIVE_STRAIGHT, REVERSE,
TANK_TURN), and the Pathfinder resolved DRIVE_TO segments into
collision-free paths.

This split baked **movement semantics into the path**:

- `DRIVE_TO` vs `DRIVE_STRAIGHT` told downstream layers *how* to
  move — that's Guidance's job.
- The constrained-ball expansion prescribed "drive to intermediate,
  drive straight to pickup, reverse back" — a movement recipe, not
  a route graph.
- The Pathfinder had to understand action types to decide which
  segments to pathfind — coupling it to the Route Builder's
  vocabulary.

**The fix:** merge Route Builder and Pathfinding into a single **Route
Compiler** that produces a flat annotated waypoint sequence with no
movement semantics. The expansion (which positions to visit) and the
collision avoidance (how to get between them without hitting obstacles)
are interleaved in a single pass. There is no intermediate
representation carrying action types.

### Layer 1 — Pickup Geometry (`pickup_geometry.py`) ✅

See `docs/fit-based-pathing.md` for full specification.

**Input:** ball positions, obstacle map, robot geometry.

**Output:** `PickupGeometryResult` — per-ball classified pickup
candidates (SAFE / IN_BETWEEN / CONSTRAINED), distance field, radii.

### Layer 2 — Route Strategy (`route_strategy.py`) ✅

See `docs/fit-based-pathing.md` for full specification.

**Input:** pickup geometry result, start pose, unload position.

**Output:** `RouteStrategyResult` — ordered `RouteStop`s with chosen
candidates, intermediate nodes, unreachable ball list.

### Layer 3 — Route Compiler (`planner.py`)

**Input:**
- `RouteStrategyResult` from layer 2
- `distance_field: np.ndarray` from layer 1 (cm per pixel, same grid)
- `start_pose: HybridPose`
- `half_width_cm: float` (robot half-width, ~9.75 cm)

**Responsibility:**

1. Walk the ordered stops. For each stop, determine the **positions
   to visit** — no actions, just positions:
   - Constrained/in-between: intermediate → pickup → intermediate
     (the return to intermediate is needed because the pickup point
     is in a constrained zone where the robot cannot turn — it must
     return to the safe zone first)
   - Safe: pickup only

2. Between each consecutive pair of positions, ensure a
   **collision-free path** using the distance field directly.

3. **Annotate** each waypoint with its purpose — nothing more:
   - `NAVIGATE` — drive through this point (A\* detour waypoints,
     intermediate nodes, pass-through positions)
   - `PICKUP` — pick up a ball here (carries ball index)
   - `UNLOAD` — unload here

4. Append unload waypoints at the end (if unload position given).

**Output:** `RoutePlan` — a flat, ordered list of annotated waypoints.

**The output carries no movement instructions.** No DRIVE_TO vs
DRIVE_STRAIGHT. No REVERSE. No TANK_TURN. Every waypoint is just a
position + heading + annotation. The Brain decides *what intent* to
emit at each point. Guidance decides *how to move* — whether to
reverse, which direction to turn, what speed to use — based on the
live pose.

#### Collision avoidance — distance field as single source of truth

The distance field computed by layer 1 (`cv2.distanceTransform`) gives
per-pixel distance to the nearest obstacle in cm. This is the **only**
data structure used for collision checking — no separate grid dilation.

**Traversability rule:** a position is traversable if
`distance_field[row, col] >= half_width_cm`. This means the robot body
(approximated as a circle of radius `half_width`) fits without touching
any obstacle.

#### v1 algorithm

For each consecutive pair of positions (A → B):

1. **Line check.** Sample the distance field along the straight line
   from A to B (Bresenham or equivalent). Compute
   `min_clearance = min(distance_field[samples])`.

2. **Direct connection** (`min_clearance >= half_width_cm`). No extra
   waypoints needed — the straight line is collision-free.

3. **A\* fallback** (`min_clearance < half_width_cm`). Run 2D grid A\*
   on the distance field:
   - A cell is passable if `distance_field[row, col] >= half_width_cm`.
   - Heuristic: Euclidean distance.
   - Step size: 1 cm (grid resolution).
   - Output: list of `(x, y)` grid positions.

4. **Path simplification.** Apply Douglas-Peucker to reduce waypoint
   count. After simplification, **re-verify** each simplified segment
   with the line check (step 1). If any segment fails, keep the
   original unsimplified waypoints for that segment.

5. **Heading assignment.** A\* waypoints have no inherent heading.
   Assign heading as the bearing to the next waypoint:
   `theta = atan2(next.y - this.y, next.x - this.x)`. The final
   waypoint in each segment inherits the heading of the destination
   position.

#### v1 scope and limitations

- **No ball avoidance.** Uncollected balls are not added to the
  obstacle map. Since v1 uses nearest-neighbor ordering, the robot
  generally moves away from remaining clusters. Ball avoidance can be
  added in v2 by inflating ball positions into the distance field
  before the line check / A\* step.
- **Circle approximation.** The robot body is approximated as a circle
  of radius `half_width`. This is conservative (the actual body is
  rectangular), but safe and simple. Oriented-rectangle checks can
  replace this later if needed.
- **No dynamic replanning.** The route is compiled once. If the field
  changes mid-run, the Brain must request a full replan from Path.

#### How intermediate round-trips work without movement semantics

Consider a constrained ball pickup. The route compiler emits:

```
... → A* waypoints → NAVIGATE(intermediate) → NAVIGATE(pickup) →
      PICKUP(pickup, ball=3) → NAVIGATE(intermediate) → A* waypoints → ...
```

The intermediate appears twice — on the way in and on the way out.
The path doesn't say "reverse." It just says "go to intermediate."

What happens at runtime:
1. **Brain** feeds waypoints to Guidance as a DRIVE step.
2. **Guidance** drives to the intermediate node (turning + driving).
3. **Guidance** drives to the pickup node. Since the pickup is ahead
   along the intermediate's heading, guidance drives forward.
4. **Brain** sees PICKUP annotation → calls commander.pickup().
5. **Brain** feeds the next waypoints (starting with intermediate
   again) to Guidance.
6. **Guidance** sees the intermediate is directly behind the robot
   (it just drove forward from there). The reverse-drive logic
   kicks in: target behind + close → drive backward. **Guidance
   decides to reverse — the path never told it to.**

This is the correct separation: Path knows the robot needs to return
to the safe zone. Guidance knows the fastest way there is to reverse.

#### What the output replaces

The current `RoutePlan` uses `id()` matching between `points` and
`pickup_poses` — the Brain's route interpreter reconstructs structure
by checking Python object identity. This is fragile and encodes
structure implicitly.

The redesigned output should carry explicit annotations so the route
interpreter can trivially split the waypoint sequence into Steps
(DRIVE → PICKUP → DRIVE → PICKUP → ... → DRIVE → UNLOAD) without
identity tricks.

Proposed output types:

```python
class WaypointKind(Enum):
    NAVIGATE = "navigate"   # drive through
    PICKUP = "pickup"       # pick up ball here
    UNLOAD = "unload"       # unload here

@dataclass(frozen=True)
class RouteWaypoint:
    x_cm: float
    y_cm: float
    theta_rad: float
    kind: WaypointKind
    ball_index: int | None = None   # which ball (for PICKUP)

@dataclass(frozen=True)
class RoutePlan:
    waypoints: tuple[RouteWaypoint, ...]
    unload_goal_cm: tuple[float, float] | None = None
```

The route interpreter splits this into Steps:
- Accumulate NAVIGATE waypoints → flush as a DRIVE step when a
  PICKUP or UNLOAD waypoint is reached (include the PICKUP/UNLOAD
  waypoint as the final waypoint of the DRIVE step so guidance
  drives all the way there).
- Emit a PICKUP or UNLOAD step after each flush.

This is the same logic the current interpreter uses, but with
explicit kind checks instead of `id()` matching.

---

## Boundary: Path → Brain

**Path outputs:** `RoutePlan` — flat annotated waypoint sequence.

**Brain consumes:** splits the sequence into `Step`s via
`interpret_route()`. Each step is either DRIVE (with waypoints) or
PICKUP or UNLOAD. The Brain walks steps sequentially, feeding DRIVE
waypoints to Guidance, calling commander for PICKUP/UNLOAD.

**The boundary contract:**
- Waypoints are in field-cm, bottom-left origin.
- Headings are radians, 0 = +X, positive = CCW.
- `WaypointKind` annotations are the only structural information.
- The sequence is ordered — visit in order, no skipping.
- A PICKUP or UNLOAD waypoint terminates a DRIVE segment.
- All waypoint-to-waypoint transitions are collision-free (the path
  guarantees this via A\* insertion).

**Path does NOT guarantee:**
- That the robot can physically reach a waypoint from the previous
  one without turning — Guidance handles turns.
- That any particular movement mode is used — Guidance chooses
  forward, reverse, or turn based on live geometry.
- Mid-route replanning — if a ball moves, the Brain must request
  a new route from Path.

---

## Naming notes

Names follow what each layer owns. Guidance is borrowed from
aerospace — precise, but Motion or Maneuver carry the same meaning
if the team reads them faster. Everything else holds.
