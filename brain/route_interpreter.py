"""Route interpretation -- converts a RoutePlan into a list of executable Steps.

Splits the flat annotated waypoint sequence on ``WaypointKind``:

- Accumulate NAVIGATE waypoints into the current DRIVE segment.
- On PICKUP or UNLOAD, include that waypoint as the final drive
  destination, flush the DRIVE step, then emit the action step.
- Remaining NAVIGATE waypoints at the end become a trailing DRIVE step.
"""

from __future__ import annotations

from path.models import HybridPose, RoutePlan, WaypointKind

from brain.models import Step, StepKind


def interpret_route(plan: RoutePlan) -> list[Step]:
    """Convert a RoutePlan into an ordered list of Steps.

    Parameters
    ----------
    plan : RoutePlan
        Flat annotated waypoint sequence from the Path layer.

    Returns
    -------
    list[Step]
        Ordered steps the Brain should execute sequentially.
    """
    if not plan.waypoints:
        return []

    steps: list[Step] = []
    current_waypoints: list[HybridPose] = []

    for wp in plan.waypoints:
        pose = HybridPose(wp.x_cm, wp.y_cm, wp.theta_rad)

        if wp.kind == WaypointKind.NAVIGATE:
            current_waypoints.append(pose)

        elif wp.kind == WaypointKind.PICKUP:
            # Include the pickup position as the final drive destination
            # so guidance drives all the way there.  Carry the pickup's
            # acceptance window onto the DRIVE step so guidance can finish
            # without a hard pivot on open balls.
            current_waypoints.append(pose)
            steps.append(Step(
                kind=StepKind.DRIVE,
                waypoints=tuple(current_waypoints),
                pickup_zone=wp.pickup_zone,
            ))
            current_waypoints = []
            steps.append(Step(
                kind=StepKind.PICKUP,
                obstacle_constrained=wp.obstacle_constrained,
            ))

        elif wp.kind == WaypointKind.UNLOAD:
            current_waypoints.append(pose)
            steps.append(Step(
                kind=StepKind.DRIVE,
                waypoints=tuple(current_waypoints),
            ))
            current_waypoints = []
            steps.append(Step(kind=StepKind.UNLOAD))

    # Flush any remaining NAVIGATE waypoints.
    if current_waypoints:
        steps.append(Step(
            kind=StepKind.DRIVE,
            waypoints=tuple(current_waypoints),
        ))

    return steps
