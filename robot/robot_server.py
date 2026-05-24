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
  collector_travel_position - Raise collector to safe driving position
  pickup_assist           - Small pipe jiggle for collecting one ball
  unload_full_cycle       - Full pipe cycle for unloading balls at the goal
  pickup                  - Compatibility alias for pickup_assist
  dropoff                 - Compatibility alias for unload_full_cycle
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
COLLECTOR_TRAVEL_UNITS = 22
COLLECTOR_TRAVEL_SPEED = 50
PICKUP_ASSIST_UNITS = 2
PICKUP_ASSIST_SPEED = 35
UNLOAD_FULL_CYCLE_UNITS = 22
UNLOAD_FULL_CYCLE_SPEED = 75

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
                        degrees=motor_deg, brake=True, block=True)
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
                            degrees=motor_deg, brake=True, block=True)
    else:
        # Turn left: right motor forward, left motor backward
        tank.on_for_degrees(left_speed=SpeedPercent(spd), right_speed=SpeedPercent(-spd),
                            degrees=motor_deg, brake=True, block=True)
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

def cmd_collector_travel_position():
    """Move collector toward its raised travel position before route following."""
    motor_deg = COLLECTOR_TRAVEL_UNITS * PIPE_DEGREES_PER_UNIT
    pipe_motor.on_for_degrees(
        speed=SpeedPercent(-COLLECTOR_TRAVEL_SPEED),
        degrees=motor_deg,
        brake=True,
        block=True
    )
    return "ok: collector travel position"

def cmd_pickup_assist():
    """Small local pipe motion for collection; never a full unload stroke."""
    units = PICKUP_ASSIST_UNITS
    speed = PICKUP_ASSIST_SPEED
    motor_deg = units * PIPE_DEGREES_PER_UNIT

    pipe_motor.on_for_degrees(
        speed=SpeedPercent(speed),
        degrees=motor_deg,
        brake=True,
        block=True
    )

    pipe_motor.on_for_degrees(
        speed=SpeedPercent(-speed),
        degrees=motor_deg,
        brake=True,
        block=True
    )

    return "ok: pipe pickup assist completed"

def cmd_unload_full_cycle():
    """Full pipe motion for unloading at the goal only."""
    units = UNLOAD_FULL_CYCLE_UNITS
    speed = UNLOAD_FULL_CYCLE_SPEED
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
    
    return "ok: pipe unload full cycle completed"


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
        elif action == "collector_travel_position":
            return cmd_collector_travel_position()
        elif action in ("pickup_assist", "pickup"):
            return cmd_pickup_assist()
        elif action in ("unload_full_cycle", "dropoff"):
            return cmd_unload_full_cycle()
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
