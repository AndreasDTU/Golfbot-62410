# Autonomous Drive Quickstart

Run from the repository root unless noted:

```bash
cd /Users/peterroland/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/DTU/4_Semester/62410_CDIO-Project/Repo
```

1. Check required files exist:

```bash
ls calibration_data.npz robot_calibration.json tools/best.pt
```

2. On the EV3, start the TCP command server:

```bash
cd <repo-on-ev3>
python3 robot/robot_server.py
```

The `--drive` path uses this TCP server for all autonomous wheel and collector
commands; no UDP wheel server is required.

3. On the laptop, start live vision without motors first:

```bash
python3 tools/topdown_object_detector.py --live --camera-index 0
```

4. Confirm in the UI:

- top-down calibration is locked
- robot pose is visible
- balls/red zones look correct
- route appears sane

5. Start autonomous drive in step mode:

```bash
python3 tools/topdown_object_detector.py --live --camera-index 0 --drive --step
```

6. During the run:

- press `n` to release each autonomous target in step mode
- press `k` to start the optional drive calibration sequence when `--drive`
  is active and robot ArUco pose is valid
- press `space` to stop wheel output
- press `q` or `Esc` to quit; shutdown sends wheel stop

7. Optional drive calibration:

- clear enough field space for a 360 degree tank turn and a 10 cm forward move
- press `k`; the app stops route control, commands the EV3 turn and move in the
  background, and keeps measuring ArUco pose in the live loop
- review the overlay for expected vs actual turn, expected vs actual distance,
  lateral drift, old values, and suggested values
- press `y` to persist/apply the suggested EV3 drive calibration values, or
  press `x` to discard/cancel

Saved values are written by the EV3 command server to
`robot/robot_drive_calibration.json`. If that file is missing or invalid, the
server falls back to the built-in robot defaults.

8. Full autonomous without per-target pauses:

```bash
python3 tools/topdown_object_detector.py --live --camera-index 0 --drive
```

Use full autonomous only after step mode has worked on the real field.
