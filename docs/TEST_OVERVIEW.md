# Test Overview

This repository has dedicated smoke tests for checkerboard undistortion and perspective warp validation.

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
