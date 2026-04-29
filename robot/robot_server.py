#!/usr/bin/env python3
"""
EV3 Command Server
Runs on the EV3 brick. Listens for movement commands over TCP.

Motor layout:
  OUTPUT_B - Left drive motor  (large)
  OUTPUT_C - Right drive motor (large)
  OUTPUT_D - Pipe lift motor   (large)

Commands:
  move <units> [speed]    - Drive forward N units
  back <units> [speed]    - Drive backward N units
  turn <degrees> [speed]  - Tank turn (positive = right, negative = left)
  pipe up <units> [speed] - Raise the pipe N units
  pipe down <units> [speed] - Lower the pipe N units
  pipe stop               - Stop pipe motor immediately
  pickup                  - Pickup a ball
  dropoff                 - Dropoff all balls
  stop                    - Stop all drive motors
  ping                    - Health check, returns 'pong'
"""

import socket
import threading
import time
from ev3dev2.motor import (
    LargeMotor, MoveTank,
    OUTPUT_B, OUTPUT_C, OUTPUT_D,
    SpeedPercent
)

HOST = '0.0.0.0'
PORT = 5555

# --- Robot physical configuration (tune these to your robot) ---
WHEEL_DIAMETER_MM = 56.0             # Diameter of drive wheels in mm
AXLE_TRACK_MM     = 252.5986772      # Center-to-center distance between drive wheels in mm
MM_PER_UNIT = 9.9664                 # 1 unit = 10mm (1cm). Adjust to recalibrate.

# Pipe motor: degrees of motor rotation per unit of pipe travel
# Tune this based on your pipe mechanism's gear ratio / spool size
PIPE_DEGREES_PER_UNIT = 45.0

# --- Motor setup ---
tank         = MoveTank(OUTPUT_B, OUTPUT_C)
pipe_motor   = LargeMotor(OUTPUT_D)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def units_to_degrees(units):
    """Convert linear travel units to drive motor degrees."""
    mm = units * MM_PER_UNIT
    circumference = 3.14159 * WHEEL_DIAMETER_MM
    return (mm / circumference) * 360.0


def turn_angle_to_motor_degrees(turn_angle_deg):
    """
    Convert a desired robot heading change (degrees) to drive motor degrees.
    Uses the arc length each wheel must travel for a tank (pivot) turn.
    """
    arc = (abs(turn_angle_deg) / 360.0) * 3.14159 * AXLE_TRACK_MM
    circumference = 3.14159 * WHEEL_DIAMETER_MM
    return (arc / circumference) * 360.0


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def cmd_move(parts, forward=True):
    if len(parts) < 2:
        return "error: move/back requires <units> [speed]"
    units = float(parts[1])
    speed = float(parts[2]) if len(parts) > 2 else 50
    motor_deg = units_to_degrees(units)
    spd = speed if forward else -speed
    tank.on_for_degrees(left_speed=SpeedPercent(-spd), right_speed=SpeedPercent(-spd),
                        degrees=motor_deg, brake=True, block=False)
    direction = "forward" if forward else "backward"
    return "ok: moved {} {} units".format(direction, units)


def cmd_turn(parts):
    if len(parts) < 2:
        return "error: turn requires <degrees> [speed]"
    angle = float(parts[1])
    speed = float(parts[2]) if len(parts) > 2 else 30
    motor_deg = turn_angle_to_motor_degrees(angle)
    spd = speed
    if angle >= 0:
        # Turn right: left motor forward, right motor backward
        tank.on_for_degrees(left_speed=SpeedPercent(-spd), right_speed=SpeedPercent(spd),
                            degrees=motor_deg, brake=True, block=False)
    else:
        # Turn left: right motor forward, left motor backward
        tank.on_for_degrees(left_speed=SpeedPercent(spd), right_speed=SpeedPercent(-spd),
                            degrees=motor_deg, brake=True, block=False)
    return "ok: turned {} degrees".format(angle)


def cmd_pipe(parts):
    """
    pipe up <units> [speed]
    pipe down <units> [speed]
    pipe stop
    """
    if len(parts) < 2:
        return "error: pipe requires up/down/stop"

    subaction = parts[1].lower()

    if subaction == "stop":
        pipe_motor.off()
        return "ok: pipe stopped"

    if subaction not in ("up", "down"):
        return "error: unknown pipe subcommand '{}'".format(subaction)

    if len(parts) < 3:
        return "error: pipe {} requires <units> [speed]".format(subaction)

    units = float(parts[2])
    speed = float(parts[3]) if len(parts) > 3 else 30
    motor_deg = units * PIPE_DEGREES_PER_UNIT

    # "up" and "down" direction depends on how your motor is mounted —
    # flip the sign on speed if the pipe moves the wrong way.
    direction = -1 if subaction == "up" else 1
    pipe_motor.on_for_degrees(
        speed=SpeedPercent(speed * direction),
        degrees=motor_deg,
        brake=True,
        block=True
    )
    return "ok: pipe {} {} units".format(subaction, units)

def cmd_pickup():
    units = 20
    speed = 35
    motor_deg = units * PIPE_DEGREES_PER_UNIT

    pipe_motor.on_for_degrees(
        speed=SpeedPercent(speed),
        degrees=motor_deg,
        brake=True,
        block=True
    )

    pipe_motor.on_for_degrees(
        speed=SpeedPercent(-100),
        degrees=motor_deg,
        brake=True,
        block=True
    )

    return "ok: pipe pickup completed"

def cmd_dropoff():
    units = 22
    speed = 75
    motor_deg = units * PIPE_DEGREES_PER_UNIT

    pipe_motor.on_for_degrees(
        speed=SpeedPercent(-speed),
        degrees=motor_deg,
        brake=True,
        block=True
    )

    time.sleep(3)

    pipe_motor.on_for_degrees(
        speed=SpeedPercent(speed),
        degrees=motor_deg,
        brake=True,
        block=True
    )
    
    return "ok: pipe pickup completed"


def cmd_stop():
    tank.off()
    return "ok: drive stopped"


# ---------------------------------------------------------------------------
# Command dispatcher
# ---------------------------------------------------------------------------

def handle_command(cmd):
    parts = cmd.strip().split()
    if not parts:
        return "error: empty command"

    action = parts[0].lower()

    try:
        if action == "ping":
            return "pong"
        elif action == "move":
            return cmd_move(parts, forward=True)
        elif action == "back":
            return cmd_move(parts, forward=False)
        elif action == "turn":
            return cmd_turn(parts)
        elif action == "pipe":
            return cmd_pipe(parts)
        elif action == "stop":
            return cmd_stop()
        elif action == "pickup":
            return cmd_pickup()
        elif action == "dropoff":
            return cmd_dropoff()
        else:
            return "error: unknown command '{}'".format(action)
    except Exception as e:
        return "error: {}".format(str(e))


# ---------------------------------------------------------------------------
# TCP server
# ---------------------------------------------------------------------------

def handle_client(conn, addr):
    print("[+] Connected: {}".format(addr))
    with conn:
        while True:
            try:
                data = conn.recv(1024)
                if not data:
                    break
                cmd = data.decode('utf-8').strip()
                print("[>] {}: {}".format(addr, cmd))
                response = handle_command(cmd)
                print("[<] {}".format(response))
                conn.sendall((response + '\n').encode('utf-8'))
            except ConnectionResetError:
                break
    print("[-] Disconnected: {}".format(addr))


def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen(5)
        print("[*] EV3 robot server listening on {}:{}".format(HOST, PORT))
        while True:
            conn, addr = s.accept()
            thread = threading.Thread(target=handle_client, args=(conn, addr))
            thread.daemon = True
            thread.start()


if __name__ == '__main__':
    main()
