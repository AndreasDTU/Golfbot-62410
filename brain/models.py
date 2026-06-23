"""Brain FSM data models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from path.models import HybridPose, SafePickupZone


class BrainState(str, Enum):
    """FSM states for the Brain controller."""

    IDLE = "IDLE"
    DRIVE = "DRIVE"
    PICKUP = "PICKUP"
    UNLOAD = "UNLOAD"
    ERROR = "ERROR"
    DONE = "DONE"


class IntentAction(str, Enum):
    """Action the Brain wants the lower layers to execute."""

    DRIVE = "DRIVE"
    PICKUP = "PICKUP"
    UNLOAD = "UNLOAD"
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

    *obstacle_constrained* is only meaningful for PICKUP steps: when True the
    ball sits against the cross/wall, so the executor backs away before raising
    the tube.

    *pickup_zone* is only meaningful for DRIVE steps that end at a SAFE pickup:
    the SAFE acceptance region handed to guidance so the robot grabs from
    wherever it lands on the ball's reach circle instead of homing to one exact
    spot.  None for constrained pickups and pure navigation.
    """

    kind: StepKind
    waypoints: tuple[HybridPose, ...] = ()
    obstacle_constrained: bool = False
    pickup_zone: SafePickupZone | None = None
