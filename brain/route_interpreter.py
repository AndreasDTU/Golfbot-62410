"""Route interpretation -- converts a RoutePlan into a list of executable Steps.

Isolated in its own module so it can later migrate to the Path layer
without touching Brain internals.
"""

from __future__ import annotations

from path.pathfinding.models import HybridPose, RoutePlan

from brain.models import Step, StepKind


def interpret_route(plan: RoutePlan) -> list[Step]:
    """Convert a RoutePlan into an ordered list of Steps.

    Walks the plan's points list, splitting at pickup poses to produce
    DRIVE and PICKUP steps.  If the plan includes an unload pose, an
    UNLOAD step is appended after the final DRIVE segment.

    Pickup poses are matched by object identity (``id()``), which is
    reliable because the planner reuses the same frozen HybridPose
    instances in both ``points`` and ``pickup_poses``.

    Unload is appended at the end rather than matched within ``points``
    because ``unload_pose`` is constructed separately from the A* path
    reconstruction and may not share identity or exact value equality
    with any point in the route.

    Parameters
    ----------
    plan : RoutePlan
        The route plan from the Path layer.

    Returns
    -------
    list[Step]
        Ordered steps the Brain should execute sequentially.
    """
    if not plan.points:
        return []

    pickup_ids: set[int] = {id(p) for p in plan.pickup_poses}

    steps: list[Step] = []
    current_waypoints: list[HybridPose] = []

    for point in plan.points:
        current_waypoints.append(point)

        if id(point) in pickup_ids:
            # Flush accumulated waypoints as a DRIVE step.
            steps.append(Step(
                kind=StepKind.DRIVE,
                waypoints=tuple(current_waypoints),
            ))
            current_waypoints = []
            steps.append(Step(kind=StepKind.PICKUP))

    # Flush any remaining waypoints after the last pickup.
    if current_waypoints:
        steps.append(Step(
            kind=StepKind.DRIVE,
            waypoints=tuple(current_waypoints),
        ))

    # Unload is always the final action when present.
    if plan.unload_pose is not None:
        steps.append(Step(kind=StepKind.UNLOAD))

    return steps
