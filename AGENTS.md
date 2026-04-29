# AGENTS.md — Vision & Robot Control Repository

Status: ACTIVE
Purpose: Define rules, contracts, and expectations for coding agents operating on this repository.
Scope: Entire repository unless overridden by a deeper AGENTS.md.

---

# Core Principles

## 1. Determinism First

All perception and control systems must behave deterministically.

Agents MUST:

* Avoid nondeterministic behavior unless explicitly required.
* Avoid hidden global state.
* Avoid race conditions.
* Prefer explicit data flow over implicit state.

Vision outputs must depend only on current input frame and configuration.

---

## 2. Debuggability is Mandatory

All perception stages must support debug visualization.

Agents MUST ensure the following debug views exist:

* Warped top-down view
* Segmentation mask
* Candidate detections
* Final filtered detections
* Grid overlay
* FPS and timing metrics

Agents MUST NOT remove debug tools unless explicitly instructed.

---

## 3. Pipeline Contract Enforcement

Agents MUST follow the vision pipeline defined in:

VISION_CV_CONTRACT.md

Pipeline order MUST remain:

1. Undistort
2. Perspective transform
3. Illumination normalization
4. Segmentation
5. Candidate extraction
6. Filtering
7. Grid mapping

Agents MUST NOT reorder these steps without explicit contract update.

---

## 4. Performance Safety

Vision must run in real time.

Minimum requirement:

* ≥ 20 FPS sustained

Agents MUST:

* Avoid unnecessary allocations inside frame loop
* Avoid expensive per-frame initialization
* Precompute static transforms
* Prefer simple algorithms over complex ones

Agents MUST profile before introducing heavy computation.

---

## 5. Contest Reliability Priority

This is a robotics contest system.

Agents MUST prioritize:

1. Reliability
2. Determinism
3. Debuggability
4. Performance
5. Elegance (lowest priority)

Simple, reliable solutions are preferred over clever or complex solutions.

---

# Vision System Rules

Agents MUST:

* Use classical computer vision as defined in VISION_CV_CONTRACT.md
* NOT introduce ML models unless explicitly instructed
* NOT introduce training pipelines
* NOT introduce external runtime dependencies without approval

Allowed libraries:

* OpenCV
* NumPy
* Standard Python libraries

Disallowed unless approved:

* PyTorch
* TensorFlow
* large ML frameworks

---

# Camera & Geometry Rules

Agents MUST treat camera calibration and homography as critical system components.

Agents MUST:

* Never silently modify homography
* Never silently recalibrate during runtime
* Expose calibration tools separately

Homography must be stable during robot operation.

---

# Safety Rules

Agents MUST ensure:

* Robot never receives undefined coordinates
* Robot never receives NaN or invalid positions
* Vision failures degrade gracefully

If detection fails:

Allowed fallback:

* Return empty detection list
* Preserve last valid detection (if safe)

Not allowed:

* Return random coordinates
* Return unvalidated data

---

# Code Modification Rules

Agents MUST:

* Make minimal necessary changes
* Preserve existing architecture
* Preserve contracts
* Preserve debug functionality

Agents MUST NOT:

* Rewrite large systems without explicit instruction
* Introduce architectural changes without justification
* Remove existing safeguards

---

# File Structure Expectations

Expected structure:

```
/vision
    calibration.py
    pipeline.py
    segmentation.py
    detection.py
    grid_mapping.py

/tools
    calibrate_camera.py
    tune_thresholds.py
    debug_viewer.py

/docs
    VISION_CV_CONTRACT.md

AGENTS.md
```

Agents MUST follow this structure unless instructed otherwise.

---

# Testing Requirements

Agents MUST ensure new features include:

* Debug visibility
* Deterministic behavior
* No performance regression

Agents SHOULD include simple validation scripts when appropriate.

---

# Failure Handling Contract

If vision cannot confidently detect balls:

System MUST:

* Report no detections

System MUST NOT:

* Fabricate detections

---

# Change Approval Rules

Agents MAY change:

* Threshold values
* Internal implementation details
* Debug visualization

Agents MUST NOT change without approval:

* Pipeline order
* External interfaces
* Output data formats
* Contract definitions

---

# Priority Order

When making decisions, agents MUST prioritize:

1. Contract correctness
2. Contest reliability
3. Debuggability
4. Determinism
5. Performance
6. Code elegance

---

# Summary

This repository implements a contest-critical real-time vision system.

Agents must act conservatively, preserve reliability, and follow contracts strictly.

Undefined or unsafe behavior is unacceptable.

When unsure, agents must choose the safest deterministic option.