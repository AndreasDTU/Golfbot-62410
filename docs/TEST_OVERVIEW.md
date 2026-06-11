# Test Overview

This repository has dedicated smoke tests for checkerboard undistortion and perspective warp validation.

## Focused Robot Control Tests

```bash
env PYTHONPYCACHEPREFIX=/private/tmp/codex-pycache python3 -m unittest test.test_drive_control test.test_robot_server_commands test.test_robot_controller_safety
```

These tests cover closed-loop drive handoff, drive calibration state ownership,
EV3 TCP command safety, drive calibration `drivecal get/set` protocol handling,
persistence, and the controller helpers that call those commands. The
`test_drive_control` module imports OpenCV through the top-down app, so run it
with the same Python environment used for live vision.

Drive calibration math and persistence helpers can also be run in isolation:

```bash
env PYTHONPYCACHEPREFIX=/private/tmp/codex-pycache python3 -m unittest test.test_drive_calibration
```

## Full Perspective + Distortion Comparison (single command)

Run this to generate one final comparison frame containing:
- Original
- Undistorted
- Warp (Original)
- Warp (Undistorted)

```bash
python /Users/alex/PycharmProjects/Golfbot-62410/tools/perspective_warp_test.py \
  --input /Users/alex/PycharmProjects/Golfbot-62410/test/images/perspective_test.png \
  --pattern-cols 8 \
  --pattern-rows 6 \
  --square-px 80 \
  --final-test
```

Primary output artifact:
- `/Users/alex/PycharmProjects/Golfbot-62410/test/artifacts/perspective_final_comparison.png`

Report output:
- `/Users/alex/PycharmProjects/Golfbot-62410/test/artifacts/perspective_report.json`

## Smoke Test (checkerboard only)

```bash
python /Users/alex/PycharmProjects/Golfbot-62410/tools/checkerboard_smoke_test.py
```
