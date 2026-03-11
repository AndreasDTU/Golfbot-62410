from ev3dev2.motor import OUTPUT_B, OUTPUT_C, SpeedPercent, MoveTank
from ev3dev2.sensor import INPUT_1, INPUT_2, INPUT_3, INPUT_4
from ev3dev2.sensor.lego import IRSensor

import numpy as np

wheel_diameter = 56
wheel_track = 200

tank_drive = MoveTank(OUTPUT_B, OUTPUT_C)

forward_speed = -100
backward_speed = 100

# Drive the robot a certain distance (in mm) forward and right (will be combined movement)
def drive(forward_distance: int, right_distance: int):
    if forward_distance == 0 and right_distance == 0:
        return
    # Only drive forward/backward?
    if right_distance == 0:
        rotation_count = abs(forward_distance / wheel_diameter)
        speed = SpeedPercent(forward_speed if forward_distance > 0 else backward_speed)
        tank_drive.on_for_rotations(speed, speed, rotation_count)
    # Combine with rotation
    else:
        theta = 2 * np.arctan2(right_distance, forward_distance)

        if np.isclose(theta, 0.0):
            rotation_count = abs(forward_distance / wheel_diameter)
            speed = SpeedPercent(forward_speed if forward_distance > 0 else backward_speed)
            tank_drive.on_for_rotations(speed, speed, rotation_count)
            return

        if np.isclose(np.sin(theta), 0.0):
            arc_radius = right_distance / (1 - np.cos(theta))
        else:
            arc_radius = forward_distance / np.sin(theta)

        left_distance = theta * (arc_radius + wheel_track / 2)
        right_wheel_distance = theta * (arc_radius - wheel_track / 2)

        left_rotations = abs(left_distance / wheel_diameter)
        right_rotations = abs(right_wheel_distance / wheel_diameter)

        max_rotations = max(left_rotations, right_rotations)
        if np.isclose(max_rotations, 0.0):
            return

        left_speed = SpeedPercent(
            (left_rotations / max_rotations) * (forward_speed if left_distance >= 0 else backward_speed)
        )
        right_speed = SpeedPercent(
            (right_rotations / max_rotations) * (forward_speed if right_wheel_distance >= 0 else backward_speed)
        )

        tank_drive.left_motor.on_for_rotations(left_speed, left_rotations, block=False)
        tank_drive.right_motor.on_for_rotations(right_speed, right_rotations, block=True)

# Turn the robot on the spot, rotation in degrees (positive = clockwise, negative = counterclockwise)
def turn(degrees: float):
    if np.isclose(degrees, 0.0):
        return

    theta = np.deg2rad(degrees)
    wheel_distance = (wheel_track / 2) * abs(theta)
    rotation_count = wheel_distance / wheel_diameter

    left_speed = SpeedPercent(forward_speed if degrees > 0 else backward_speed)
    right_speed = SpeedPercent(backward_speed if degrees > 0 else forward_speed)

    tank_drive.left_motor.on_for_rotations(left_speed, rotation_count, block=False)
    tank_drive.right_motor.on_for_rotations(right_speed, rotation_count, block=True)