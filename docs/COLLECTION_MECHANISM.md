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

## Motor Commands

The pipe motor has two distinct software commands:

- `pickup_assist()`: a small local back-and-forth pipe motion used during
  autonomous ball collection. This assists engagement with the retaining
  mechanism and must not run a full pipe stroke.
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

## Safety Rule

Full pipe motion is allowed only in an unloading state at the goal. Collection,
including orange/VIP ball collection, uses the same small assist command as
ordinary ball collection. If route state, pose, or pickup state is ambiguous, the
safe behavior is to stop wheel output and avoid actuator motion until the state
is valid again.
