from ev3dev2.motor import LargeMotor, OUTPUT_D, SpeedPercent

motor = LargeMotor(OUTPUT_D)

gear_ratio = (40.0 / 12.0) * (36.0 / 12.0)

# Pickup a ball right in front of the robot
def pickup():
    pipe_degrees = 30
    motor_degrees = pipe_degrees / gear_ratio
    motor.on_for_degrees(SpeedPercent(100), motor_degrees)
    motor.on_for_degrees(SpeedPercent(-100), motor_degrees)

# Drop all the balls out the back
def drop_all():
    pipe_degrees = 60
    motor_degrees = pipe_degrees / gear_ratio
    motor.on_for_degrees(SpeedPercent(-100), motor_degrees)

def reset_after_drop():
    pipe_degrees = 60
    motor_degrees = pipe_degrees / gear_ratio
    motor.on_for_degrees(SpeedPercent(100), motor_degrees / 2.0)
    motor.on_for_degrees(SpeedPercent(50), motor_degrees / 2.0)