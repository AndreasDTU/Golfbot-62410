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

The detector drive loop no longer commands the collector travel position before
route following. The operator is responsible for placing the collector in its
safe travel position before starting autonomous drive. After `pickup_assist()`
completes, the software still records the collector state as `TRAVEL` for local
state bookkeeping.

Autonomous route-following wheel commands and near-zone pickup actions all use
the EV3 TCP command server. The drive loop does not use UDP transport.

## Motor Commands

The pipe motor has three distinct software commands:

- `collector_travel_position()`: raises or keeps the collector in its safe
  driving position. This remains available for manual/operator use, but the
  autonomous drive loop does not call it.
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

`tools/collector_playground.py` is a standalone GUI and terminal playground for
manual collector/pipe testing before autonomous runs. It connects only to the
EV3 TCP command interface through `robot/controller.py`; it does not start wheel
movement, route following, the vision pipeline, the route planner, or ball
detection.

GUI mode is the default. Start it from the repository root while
`robot/robot_server.py` is running on the EV3:

```text
python3 tools/collector_playground.py --host <EV3_IP>
```

To preview the GUI and exercise the command/state flow without an EV3 or robot
server, use dummy mode:

```text
python3 tools/collector_playground.py --dummy
```

Use terminal REPL mode when a display is unavailable:

```text
python3 tools/collector_playground.py --cli --host <EV3_IP>
```

Useful options:

```text
--port 5555
--timeout 15
--max-manual-units 5
--no-confirm
--dummy
```

The GUI uses OpenCV HighGUI, like `tools/pathfinding_sandbox.py`, so it avoids
native `tkinter`/Tk compatibility issues. It shows the configured host, port,
and timeout from the CLI and provides `Connect` and `Disconnect` controls.
Collector command buttons are disabled until a TCP connection is established,
except in `--dummy` mode where the GUI starts ready with a local no-network
controller. Use `+`/`-` or the unit buttons to adjust the manual movement
amount. The robot drawing shows a simple side profile and top view: the body,
wheels, front direction, and collector pipe. Pipe color and position change
with the current software belief (`UNKNOWN`, `TRAVEL`, `PICKUP_ASSIST`,
`UNLOADING`, `MANUAL_UP`, `MANUAL_DOWN`, or `STOPPED`). This visualization is
only an open-loop belief display; it is not sensor feedback.

Available GUI buttons and equivalent terminal commands:

- `travel`: command `collector_travel_position()` for raised/safe driving
  height (`Travel Position`).
- `assist` / `pickup`: command `pickup_assist()` for the small collection
  motion (`Pickup Assist`).
- `unload` / `dropoff`: command `unload_full_cycle()` for the full unloading
  stroke (`Unload Full Cycle`); this asks for confirmation unless
  `--no-confirm` or `--yes` is used.
- `up <units>` / `down <units>`: manually raise/lower the pipe by a bounded
  open-loop amount (`Pipe Up` / `Pipe Down`).
- `stop`: stop the pipe motor when supported by the EV3 server (`Stop Pipe`).
- `status`: print the current software belief in terminal mode.
- `help`, `quit`, `exit`: show terminal help or exit after a pipe-stop attempt.

There is no collector position sensor. The playground state is only a software
belief such as `UNKNOWN`, `TRAVEL`, `PICKUP_ASSIST`, or `UNLOADING`; it is not a
verified physical height. Recommended startup procedure:

1. Physically place the collector in a known raised/travel position.
2. Start `tools/collector_playground.py`.
3. Connect/configure the EV3 host and port.
4. Press `Travel Position`.
5. Test small bounded `Pipe Up` / `Pipe Down` movements.
6. Use `Unload Full Cycle` only deliberately when the robot is physically clear
   for a full stroke.

## Safety Rule

Full pipe motion is allowed only in an `UNLOADING` state at the goal.
Collection, including orange/VIP ball collection, uses the same small assist
command as ordinary ball collection. Route following must not run while the
collector is known or suspected to be fully lowered. If route state, pose,
pickup state, or collector height is ambiguous, the safe behavior is to stop
wheel output and avoid route-following motion until the collector has been
commanded back to `TRAVEL`.
