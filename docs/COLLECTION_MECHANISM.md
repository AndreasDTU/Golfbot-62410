# Collection Mechanism

Status: ACTIVE HARDWARE CONTRACT, AUTONOMOUS INTEGRATION PAUSED

## Physical Construction

The robot collects table-tennis balls with a vertical tube mounted at the front
of the robot. During a planned pickup, the planner should place the robot so the
tube center is above the target ball. The final near-zone move should be a
straight robot body-center move that ends with the tube center on the ball
coordinate.

The pickup tube lowers vertically over the ball. When the ball enters the tube,
a one-way retaining mechanism traps it so it cannot fall back out through the
pickup end.

The opposite end of the tube is the unloading/output end. Because balls enter
through the pickup end and leave through the opposite end, storage and unloading
follow FIFO order: the first ball collected is the first ball unloaded.

## Travel Height

During route following, the collector must be in a raised travel position. In
this position the tube/snout is high enough above the field surface that the
robot can drive, turn, and position the tube over a ball without scraping the
field, pushing balls unintentionally, or colliding with obstacles.

There is currently no complete autonomous drive loop in this checkout. Until a
new Brain/Guidance/Control stack is rebuilt and tested, the operator is
responsible for placing the collector in its safe travel position before any
manual robot movement.

## Motor Commands

The pipe motor has three distinct software commands exposed by
`control/controller.py` and handled by `control/robot/robot_server.py`:

- `collector_travel_position()`: raises or keeps the collector in its safe
  driving position.
- `pickup_assist()`: a small local back-and-forth pipe motion intended for ball
  collection. It must not run a full pipe stroke.
- `unload_full_cycle()`: the full pipe motion used only when unloading balls at
  the goal.

Compatibility aliases exist in the controller/server for older operator habits:

- `pickup` maps to `pickup_assist`
- `dropoff` maps to `unload_full_cycle`

New code should use the explicit command names.

## Manual Collector Playground

The manual playground lives at:

```text
control/tools/collector_playground.py
```

It is a standalone GUI and terminal playground for manual collector/pipe
testing. It connects only to the EV3 TCP command interface through
`control/controller.py`; it does not start route following, the vision pipeline,
the route planner, or ball detection.

Typical commands from the repository root:

```bash
PYTHONPATH=. python3 control/tools/collector_playground.py --host <EV3_IP>
PYTHONPATH=. python3 control/tools/collector_playground.py --dummy
PYTHONPATH=. python3 control/tools/collector_playground.py --cli --host <EV3_IP>
```

The GUI uses OpenCV HighGUI. Its displayed collector state is only a software
belief such as `UNKNOWN`, `TRAVEL`, `PICKUP_ASSIST`, or `UNLOADING`; it is not
sensor feedback.

Recommended startup procedure:

1. Physically place the collector in a known raised/travel position.
2. Start the EV3 server: `PYTHONPATH=. python3 control/robot/robot_server.py`.
3. Start `control/tools/collector_playground.py`.
4. Connect/configure the EV3 host and port.
5. Press `Travel Position`.
6. Test small bounded `Pipe Up` / `Pipe Down` movements.
7. Use `Unload Full Cycle` only deliberately when the robot is physically clear
   for a full stroke.

## Safety Rule

Full pipe motion is allowed only in an unloading-at-goal state. Collection,
including orange/VIP ball collection, uses the same small assist command as
ordinary ball collection. Route following must not run while the collector is
known or suspected to be fully lowered. If route state, pose, pickup state, or
collector height is ambiguous, the safe behavior is to stop wheel output and
avoid route-following motion until the collector has been commanded back to
`TRAVEL`.
