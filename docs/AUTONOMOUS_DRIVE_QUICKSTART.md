# Autonomous Drive Quickstart

Status: PAUSED DURING MOVEMENT REWORK

The previous autonomous entrypoint, `tools/topdown_object_detector.py`, has been
deleted. There is currently no complete laptop-side autonomous app that can run
`--live`, `--drive`, or `--step`.

Do not use this document as a claim that autonomous driving is currently
available. It records the safe current state and the pieces that still exist.

## Existing Hardware Pieces

The EV3 TCP command server still exists:

```text
control/robot/robot_server.py
```

Run it from the repository root on the EV3 so imports resolve correctly:

```bash
PYTHONPATH=. python3 control/robot/robot_server.py
```

The server supports movement and collector commands such as:

```text
move <units> [speed]
back <units> [speed]
turn <degrees> [speed]
LR <left> <right>
drivecal get
drivecal set <axle_track_mm> <mm_per_unit>
collector_travel_position
pickup_assist
unload_full_cycle
stop
ping
```

Drive calibration values are persisted beside the EV3 server as:

```text
control/robot/robot_drive_calibration.json
```

if the server creates or updates that file.

## Existing Manual Tools

Manual collector testing is still available through:

```bash
PYTHONPATH=. python3 control/tools/collector_playground.py --host <EV3_IP>
PYTHONPATH=. python3 control/tools/collector_playground.py --dummy
PYTHONPATH=. python3 control/tools/collector_playground.py --cli --host <EV3_IP>
```

The helper imports are cleaned for the new layer layout. Keep `PYTHONPATH=.`
until these scripts are converted into proper package entrypoints.

## Autonomous Rebuild Prerequisites

Before restoring an autonomous quickstart, these layers need real contracts and
hardware gates from `docs/refactor/movement_rework_integration_plan.md`:

1. Control command API with sim/real backend and boundary logging.
2. Localization pose plus freshness/validity boundary.
3. Guidance conversion from intent plus live pose to `turn` / `drive` /
   `adjust`.
4. Brain/FSM route cursor and arbitration.

Only after those gates pass should this quickstart describe a full autonomous
run again.
