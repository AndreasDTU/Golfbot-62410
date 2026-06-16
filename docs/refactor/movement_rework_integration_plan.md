# Staged integration & test plan — robot movement rework

Companion to the layer architecture proposal. Build order is bottom-up; each layer is proven on the real robot and frozen before the next one is built.

## Principle

The goal is **failure isolation by construction**: at every step, everything below the layer you are working on is already proven on the robot and frozen, so the only unproven thing in the stack is the layer you just added. A failure can therefore only be that layer or its wiring to the one below.

Two mechanisms make this hold:

- **Temporary driver.** A stand-in sits in the slot where the next layer up will go, so each layer can be exercised while it is the top of the stack. Each driver is replaced by the real layer at the next stage.
- **Boundary logging.** Every layer logs its input and output from day one. Isolation tells you *it's the new layer*; the logs tell you *which boundary* first emitted something wrong once the stack is tall.

## Ground rules

- **Freeze means freeze.** Isolation only holds if the base is stable. If a lower layer is changed, re-run its gate before trusting anything above it.
- **The robot is the gate; the sim is the iteration loop.** Kill logic bugs in the simulated backend so that hardware sessions only test what the robot alone can reveal (traction, real latency, motor behavior).
- **A gate produces a spec, not just a pass.** Each stage's measured numbers become the next layer's design assumptions. Every layer is built against the *measured* behavior of the one below it, never an assumption.

## Before Stage 1 (cross-cutting setup)

These must exist before the first gate, because every stage depends on them:

- **Units and frames contract**, written down once: cm vs mm, radians vs degrees, where heading zero points, positive rotation direction. Every new boundary is a chance for a sign flip.
- **Boundary logging** wired into each layer interface (timestamped input + output).
- **Sim/real backend** behind the Control command API, so the whole stack can run without the EV3.
- **Decided: non-blocking (ticking) commands.** All movement commands (`turn`, `drive`, `adjust`) are non-blocking and produce one wheel-speed message per call. Guidance runs every frame, reads the live pose, and calls the appropriate Control command each tick. The six-layer split holds as drawn. Only pickup/unload actuator protocols use blocking calls. *(Decision recorded 16 Jun 2026.)*

---

## Stage 1 — Control

**Build:** the execution layer only. Translates `turn` / `drive` / `adjust` into EV3 motor output; hosts the safety/validity gate and the sim/real backend.

**Driver above:** a manual REPL or keyboard teleop — you issue `turn 30`, `drive 30`, `adjust 3` directly.

**Test in isolation:** sim backend for command/units sanity; then the bench with the real robot.

**Hardware gate (defaults — refine from calibration data):**
- `turn(θ)`: |measured − θ| ≤ 3° over 10 trials, including angles near the ±180° wrap.
- `drive(d)`: |measured − d| ≤ 2 cm over 10 trials.
- `adjust(small θ)`: |measured − θ| ≤ 3°.
- Repeatability (std dev) recorded: turn ≤ 2°, drive ≤ 1.5 cm.
- Command-to-motion latency: ≤ 50 ms (measured and recorded).

**Hands upward:** measured accuracy *and* repeatability *and* latency. These become Guidance's correction budget.

**Freeze on pass:** yes.

---

## Stage 1b — Localization (in parallel with Stage 1)

**Build:** detections → clean `RobotPose` (heading, smoothing, freshness/validity flag). Independent of the motor stack, so it can be built alongside Control.

**Driver above:** none needed — drive the robot by hand (or via Stage-1 Control once it passes) and compare reported pose to ground truth.

**Test in isolation:** replay recorded detection streams; static and moving comparisons against measured ground truth.

**Hardware gate (defaults — refine from calibration data):**
- Static pose error ≤ 1.5 cm / 3°.
- Tracking error during motion ≤ 3 cm / 5°.
- End-to-end latency: ≤ 80 ms (recorded).
- Validity flag asserts within 3 frames of detection loss, and clears correctly on reacquire.

**Hands upward:** pose accuracy and latency numbers — the inputs Guidance closes the loop on.

**Must pass before Stage 2:** Guidance is the first layer that closes the loop on live pose; if localization is laggy or noisy, Guidance will look broken when the real fault is sensing.

**Freeze on pass:** yes.

---

## Stage 2 — Guidance

**Build:** intent + live pose → `turn` / `drive` / `adjust` commands (the geometry). First layer to close the loop. Built as a **clean new module** (`guidance/`); the old `route_tracking.py` code using the legacy `steer()` API is deprecated and not carried forward. Guidance must use `RobotCommander.turn()` / `.drive()` / `.adjust()` / `.stop()` exclusively — no direct wheel-speed or `steer()` calls.

**Architecture decision (resolved):** non-blocking ticking. Guidance runs every frame, reads the live pose, and calls one Control command per tick. The six-layer split holds. Only pickup/unload actuator protocols use blocking calls.

**Driver above:** hardcoded path segments — a short list of `(x, y, θ)` waypoints compiled into the test, not generated by the Path layer. This provides a known-good route for isolation testing without depending on the Path or Brain layers.

**Test in isolation:**
1. Sim backend with synthetic start/goal pairs (fast iteration, logic bugs).
2. Robot with hardcoded path segments: straight line, 90° turn, L-shaped two-segment path, and a short multi-waypoint route. Camera provides live pose; Guidance closes the loop.

**Hardware gate (defaults — refine from calibration data):**
- From 5 random start poses, final pose error ≤ 3 cm / 5°.
- Settles within 10 s with no oscillation / limit cycle.
- Holds the above given the *measured* Control error (±3° from Stage 1) and Localization latency (≤ 80 ms from Stage 1b).

**Hands upward:** closed-loop arrival accuracy — the precision the Brain can assume at each waypoint.

**Freeze on pass:** yes.

---

## Stage 3 — Brain (FSM)

**Build:** pose + route → intent. Owns the route-progress cursor and arbitration between modes (follow, turn-to-waypoint, pickup, unload, recover).

**Driver above:** the existing Path layer, or a scripted route. At this stage the Brain quietly becomes the real top of the stack.

**Test in isolation:** sim backend driving full routes; assert intent transitions from the boundary logs.

**Hardware gate (defaults — refine from calibration data):**
- Completes a 5-waypoint route end-to-end.
- Intent transitions correct and log-verified (no implicit ordering).
- Recovers from injected faults — stale pose, missed waypoint, off-route — each returning to a defined safe/resume state.

**Hands upward:** a sequenced, fault-tolerant route executor.

**Freeze on pass:** yes.

---

## Stage 4 — Full autonomous

**Build:** nothing new — remove the last scaffold (Path drives the Brain for real).

**Hardware gate (defaults — refine from calibration data):**
- Collects 5 balls autonomously across 3 runs.
- Success rate ≥ 80%.
- No unattributed failures: any failure is traceable to a layer via the boundary logs.

---

## Reading a failure

Two questions, two mechanisms:

- *Which layer?* — During build, isolation answers it: the fault is the layer under test, because everything below is frozen and proven.
- *Which boundary, in a full run?* — Replay the boundary logs and find the first layer whose output diverged from what its input should have produced.

Isolation by build order, plus attribution by logging, is the direct answer to "we never knew what was failing."
