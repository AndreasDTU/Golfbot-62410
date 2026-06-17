"""Brain FSM controller -- route execution and mode arbitration.

The Brain receives a RoutePlan, converts it to Steps via the route
interpreter, then executes them sequentially.  Each tick advances the
FSM and returns the current BrainState.
"""

from __future__ import annotations

import math
import time

from control.commander import RobotCommander
from control.telemetry import log_event
from guidance.guidance import GuidanceController, GuidanceStatus
from localization.localization import normalize_angle
from localization.models import RobotPose
from path.pathfinding.models import RoutePlan

from brain.models import BrainIntent, BrainState, IntentAction, StepKind
from brain.route_interpreter import interpret_route

UNLOAD_SETTLE_DURATION_S = 2.0
UNLOAD_HEADING_TOLERANCE_RAD = math.radians(3.0)


class BrainController:
    """Tick-based FSM that executes a RoutePlan step by step.

    The Brain does NOT own GuidanceController or RobotCommander -- they
    are injected at construction.  Each call to ``tick()`` advances the
    FSM by one frame and returns the current ``BrainState``.

    Usage::

        brain = BrainController(guidance, commander)
        brain.load_route(plan)
        while True:
            state = brain.tick(pose, dt_s)
            if state == BrainState.DONE:
                break
    """

    def __init__(
        self,
        guidance: GuidanceController,
        commander: RobotCommander,
    ) -> None:
        self._guidance = guidance
        self._commander = commander

        self._state: BrainState = BrainState.IDLE
        self._steps: list = []
        self._step_cursor: int = 0
        self._intent: BrainIntent = BrainIntent(action=IntentAction.STOP)
        self._error_message: str = ""

        # Unload settle state
        self._unload_settle_start: float | None = None
        self._unload_heading_samples: list[float] = []
        self._unload_target_heading: float | None = None
        self._unload_corrected: bool = False

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def state(self) -> BrainState:
        """Current FSM state (also returned by ``tick``)."""
        return self._state

    @property
    def intent(self) -> BrainIntent:
        """Most recent intent emitted by the Brain."""
        return self._intent

    @property
    def step_cursor(self) -> int:
        """Index of the current step being executed."""
        return self._step_cursor

    @property
    def step_count(self) -> int:
        """Total number of steps in the loaded plan."""
        return len(self._steps)

    @property
    def error_message(self) -> str:
        """Human-readable description of the last error."""
        return self._error_message

    # ------------------------------------------------------------------
    # Route loading
    # ------------------------------------------------------------------

    def load_route(self, plan: RoutePlan) -> None:
        """Interpret a RoutePlan into steps and prepare for execution.

        Resets the FSM to IDLE and the step cursor to 0.
        """
        self._steps = interpret_route(plan)
        self._step_cursor = 0
        self._state = BrainState.IDLE
        self._error_message = ""
        self._intent = BrainIntent(action=IntentAction.STOP)
        self._reset_unload_state()
        self._guidance.clear_route()
        log_event("BRAIN", "route loaded", steps=len(self._steps))

    def reset(self) -> None:
        """Clear all state and return to IDLE with no plan."""
        self._steps = []
        self._step_cursor = 0
        self._state = BrainState.IDLE
        self._error_message = ""
        self._intent = BrainIntent(action=IntentAction.STOP)
        self._reset_unload_state()
        self._guidance.clear_route()
        log_event("BRAIN", "reset")

    def _reset_unload_state(self) -> None:
        self._unload_settle_start = None
        self._unload_heading_samples = []
        self._unload_target_heading = None
        self._unload_corrected = False

    # ------------------------------------------------------------------
    # Per-frame tick
    # ------------------------------------------------------------------

    def tick(self, pose: RobotPose | None, dt_s: float) -> BrainState:
        """Advance the FSM by one frame.

        Parameters
        ----------
        pose : RobotPose or None
            Current robot pose from localization.
        dt_s : float
            Seconds since the previous tick.

        Returns
        -------
        BrainState
            The FSM state after this tick.
        """
        if self._state == BrainState.DONE:
            return self._state

        if self._state == BrainState.ERROR:
            return self._state

        if self._state == BrainState.IDLE:
            return self._tick_idle()

        if self._state == BrainState.DRIVE:
            return self._tick_drive(pose, dt_s)

        if self._state == BrainState.PICKUP:
            return self._tick_pickup()

        if self._state == BrainState.UNLOAD:
            return self._tick_unload(pose)

        return self._state

    # ------------------------------------------------------------------
    # State handlers
    # ------------------------------------------------------------------

    def _tick_idle(self) -> BrainState:
        """Dispatch the next step or transition to DONE."""
        if self._step_cursor >= len(self._steps):
            self._state = BrainState.DONE
            self._intent = BrainIntent(action=IntentAction.STOP)
            log_event("BRAIN", "DONE", reason="all steps completed")
            return self._state

        step = self._steps[self._step_cursor]

        if step.kind == StepKind.DRIVE:
            self._guidance.set_route(list(step.waypoints))
            target = step.waypoints[-1] if step.waypoints else None
            self._intent = BrainIntent(
                action=IntentAction.DRIVE,
                target_pose=target,
            )
            self._state = BrainState.DRIVE
            log_event(
                "BRAIN", "DRIVE started",
                step=self._step_cursor,
                waypoints=len(step.waypoints),
            )

        elif step.kind == StepKind.PICKUP:
            self._intent = BrainIntent(action=IntentAction.PICKUP)
            self._state = BrainState.PICKUP
            log_event("BRAIN", "PICKUP started", step=self._step_cursor)

        elif step.kind == StepKind.UNLOAD:
            self._reset_unload_state()
            # Find target heading from the preceding DRIVE step's last waypoint.
            for i in range(self._step_cursor - 1, -1, -1):
                prev = self._steps[i]
                if prev.kind == StepKind.DRIVE and prev.waypoints:
                    self._unload_target_heading = prev.waypoints[-1].theta_rad
                    break
            self._intent = BrainIntent(action=IntentAction.UNLOAD)
            self._state = BrainState.UNLOAD
            log_event("BRAIN", "UNLOAD started", step=self._step_cursor)

        return self._state

    def _tick_drive(self, pose: RobotPose | None, dt_s: float) -> BrainState:
        """Tick guidance and check for arrival or error."""
        status = self._guidance.tick(pose, dt_s)

        if status == GuidanceStatus.ARRIVED:
            self._step_cursor += 1
            self._state = BrainState.IDLE
            log_event("BRAIN", "DRIVE complete", step=self._step_cursor - 1)
            return self._state

        if status == GuidanceStatus.ERROR:
            self._error_message = "Guidance reported ERROR during DRIVE"
            self._state = BrainState.ERROR
            self._intent = BrainIntent(action=IntentAction.STOP)
            log_event(
                "BRAIN", "ERROR",
                source="guidance", step=self._step_cursor,
            )
            return self._state

        # RUNNING or NO_POSE — stay in DRIVE.
        # Guidance already sends stop when pose is lost.
        return self._state

    def _tick_pickup(self) -> BrainState:
        """Execute blocking pickup and advance."""
        try:
            result = self._commander.pickup()
            log_event(
                "BRAIN", "PICKUP complete",
                result=result, step=self._step_cursor,
            )
        except Exception as exc:
            self._error_message = f"Pickup failed: {exc}"
            self._state = BrainState.ERROR
            self._intent = BrainIntent(action=IntentAction.STOP)
            log_event(
                "BRAIN", "ERROR",
                source="pickup", step=self._step_cursor, error=str(exc),
            )
            return self._state

        self._step_cursor += 1
        self._state = BrainState.IDLE
        return self._state

    def _tick_unload(self, pose: RobotPose | None) -> BrainState:
        """Three-phase unload: settle → heading correct → dump.

        Phase 1 (SETTLE): Accumulate heading samples for 2 seconds while
        the robot is stopped, averaging out localization noise.

        Phase 2 (CORRECT): Compare the averaged heading to the expected
        heading from the preceding DRIVE step.  If the error exceeds a
        tight tolerance, issue a single tank-turn correction.

        Phase 3 (DUMP): Call the blocking ``commander.dropoff()``.
        """
        # Phase 1: Settle — collect heading samples.
        if not self._unload_corrected:
            if self._unload_settle_start is None:
                self._unload_settle_start = time.perf_counter()
                self._unload_heading_samples = []
                log_event("BRAIN", "UNLOAD settle started", step=self._step_cursor)

            if pose is not None:
                self._unload_heading_samples.append(pose.heading_rad)

            elapsed = time.perf_counter() - self._unload_settle_start
            if elapsed < UNLOAD_SETTLE_DURATION_S:
                return self._state

            # Phase 2: Correct heading if needed.
            self._unload_corrected = True
            if (
                self._unload_target_heading is not None
                and self._unload_heading_samples
            ):
                avg_heading = self._circular_mean(self._unload_heading_samples)
                heading_error = normalize_angle(
                    self._unload_target_heading - avg_heading
                )
                log_event(
                    "BRAIN", "UNLOAD settle done",
                    samples=len(self._unload_heading_samples),
                    avg_heading_deg=math.degrees(avg_heading),
                    target_deg=math.degrees(self._unload_target_heading),
                    error_deg=math.degrees(heading_error),
                )
                if abs(heading_error) > UNLOAD_HEADING_TOLERANCE_RAD:
                    self._commander.turn(math.degrees(heading_error))
                    log_event(
                        "BRAIN", "UNLOAD heading corrected",
                        turn_deg=math.degrees(heading_error),
                    )
            return self._state

        # Phase 3: Dump.
        try:
            result = self._commander.dropoff()
            log_event(
                "BRAIN", "UNLOAD complete",
                result=result, step=self._step_cursor,
            )
        except Exception as exc:
            self._error_message = f"Unload failed: {exc}"
            self._state = BrainState.ERROR
            self._intent = BrainIntent(action=IntentAction.STOP)
            log_event(
                "BRAIN", "ERROR",
                source="unload", step=self._step_cursor, error=str(exc),
            )
            return self._state

        self._step_cursor += 1
        self._state = BrainState.IDLE
        return self._state

    @staticmethod
    def _circular_mean(angles_rad: list[float]) -> float:
        """Compute the circular (angular) mean of a list of angles."""
        sin_sum = sum(math.sin(a) for a in angles_rad)
        cos_sum = sum(math.cos(a) for a in angles_rad)
        return math.atan2(sin_sum, cos_sum)

    # ------------------------------------------------------------------
    # Error recovery
    # ------------------------------------------------------------------

    def recover(self) -> BrainState:
        """Attempt to recover from ERROR state.

        Returns to IDLE so the next tick retries the current step.
        The step cursor does not advance — the failed step is retried.
        """
        if self._state != BrainState.ERROR:
            return self._state
        self._state = BrainState.IDLE
        self._error_message = ""
        log_event("BRAIN", "recovered to IDLE")
        return self._state
