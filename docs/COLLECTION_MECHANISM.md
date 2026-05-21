# Collection Mechanism

Status: ACTIVE

## Physical Construction

The robot collects table-tennis balls with a vertical tube mounted at the front
of the robot. During a planned pickup, the route planner places the robot so the
tube center is above the target ball. The final near-zone TCP move is a straight
robot body-center move that ends with the tube center on the ball coordinate.

The pickup tube lowers vertically over the ball. When the ball enters the tube,
a one-way retaining mechanism traps it so it cannot fall back out through the
pickup end.

The opposite end of the tube is the unloading/output end. Because balls enter
through the pickup end and leave through the opposite end, storage and unloading
follow FIFO order: the first ball collected is the first ball unloaded.

## Travel Height

During normal autonomous driving and route following, the collector must be in a
raised travel position. In this position the tube/snout is high enough above the
field surface that the robot can drive, turn, and position the tube over a ball
without scraping the field, pushing balls unintentionally, or colliding with
obstacles.

The detector drive loop treats the collector position as a safety gate. Before
wheel route-following is allowed, `tools/topdown_object_detector.py` sends
`collector_travel_position()` when the collector state is unknown. After
`pickup_assist()` completes, the software returns the collector state to
`TRAVEL` before route following continues.

## Motor Commands

The pipe motor has three distinct software commands:

- `collector_travel_position()`: raises or keeps the collector in its safe
  driving position. Autonomous driving should assert this state before sending
  route-following wheel commands.
- `pickup_assist()`: a small local back-and-forth pipe motion used during
  autonomous ball collection. This assists engagement with the retaining
  mechanism and must not run a full pipe stroke or leave the collector fully
  lowered.
- `unload_full_cycle()`: the full pipe motion used only when unloading balls at
  the goal.

The autonomous collection path in
`tools/topdown_object_detector.py` may call only `pickup_assist()` after the
near-zone `turn(...)` and `move(...)` handoff. It must never call
`unload_full_cycle()` while collecting white or orange balls.

Compatibility aliases exist in `robot/controller.py` and `robot/robot_server.py`
for older operator scripts:

- `pickup` maps to `pickup_assist`
- `dropoff` maps to `unload_full_cycle`

New code should use the explicit names.

## Manual Collector Playground

`tools/collector_playground.py` is a standalone terminal playground for manual
collector/pipe testing before autonomous runs. It connects only to the EV3 TCP
command interface through `robot/controller.py`; it does not start wheel
movement, route following, the vision pipeline, the route planner, or ball
detection.

Start it from the repository root while `robot/robot_server.py` is running on
the EV3:

```text
python3 tools/collector_playground.py --host <EV3_IP>
```

Useful options:

```text
--port 5555
--timeout 15
--max-manual-units 5
--no-confirm
```

Available commands:

- `travel`: command `collector_travel_position()` for raised/safe driving
  height.
- `assist` / `pickup`: command `pickup_assist()` for the small collection
  motion.
- `unload` / `dropoff`: command `unload_full_cycle()` for the full unloading
  stroke; this asks for confirmation unless `--no-confirm` or `--yes` is used.
- `up <units>` / `down <units>`: manually raise/lower the pipe by a bounded
  open-loop amount.
- `stop`: stop the pipe motor when supported by the EV3 server.
- `status`: print the current software belief.
- `help`, `quit`, `exit`: show help or exit after a pipe-stop attempt.

There is no collector position sensor. The playground state is only a software
belief such as `UNKNOWN`, `TRAVEL`, `PICKUP_ASSIST`, or `UNLOADING`; it is not a
verified physical height. Before testing, place the collector in a known safe
position, power the EV3 server, start the playground, run `status`, then use
small bounded `up`/`down` commands or `travel` to confirm movement direction.
Do not run `unload` unless the robot is physically clear for a full stroke.

## Safety Rule

Full pipe motion is allowed only in an `UNLOADING` state at the goal.
Collection, including orange/VIP ball collection, uses the same small assist
command as ordinary ball collection. Route following must not run while the
collector is known or suspected to be fully lowered. If route state, pose,
pickup state, or collector height is ambiguous, the safe behavior is to stop
wheel output and avoid route-following motion until the collector has been
commanded back to `TRAVEL`.
