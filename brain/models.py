"""Brain FSM data models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from path.pathfinding.models import HybridPose


class BrainState(str, Enum):
    """FSM states for the Brain controller."""

    IDLE = "IDLE"
    DRIVE = "DRIVE"
    PICKUP = "PICKUP"
    UNLOAD = "UNLOAD"
    VICTORY = "VICTORY"
    ERROR = "ERROR"
    DONE = "DONE"


class IntentAction(str, Enum):
    """Action the Brain wants the lower layers to execute."""

    DRIVE = "DRIVE"
    PICKUP = "PICKUP"
    UNLOAD = "UNLOAD"
    VICTORY = "VICTORY"
    STOP = "STOP"


@dataclass(frozen=True)
class BrainIntent:
    """Intent emitted by the Brain each tick.

    The Brain does not own movement execution -- it emits an intent
    describing what it wants to happen.  Guidance and Control interpret it.
    """

    action: IntentAction
    target_pose: HybridPose | None = None


class StepKind(str, Enum):
    """Discriminator for route execution steps."""

    DRIVE = "DRIVE"
    PICKUP = "PICKUP"
    UNLOAD = "UNLOAD"


@dataclass(frozen=True)
class Step:
    """One discrete action in the route execution plan.

    For DRIVE steps, *waypoints* is the list of HybridPose waypoints
    to feed to GuidanceController.  For PICKUP/UNLOAD, waypoints is empty.
    """

    kind: StepKind
    waypoints: tuple[HybridPose, ...] = ()
