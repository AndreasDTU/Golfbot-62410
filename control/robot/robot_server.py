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
  turn <degrees> [speed]  - Tank turn (positive = CCW/left, negative = CW/right)
  pipe up <units> [speed] - Raise the pipe N units
  pipe down <units> [speed] - Lower the pipe N units
  pipe stop               - Stop pipe motor immediately
  LR <left> <right>       - Set continuous left/right wheel speeds
  drivecal get            - Read active drive calibration constants
  drivecal set <axle_track_mm> <mm_per_unit> - Persist/apply drive calibration
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
import math
import sys
from pathlib import Path
from ev3dev2.motor import (
    LargeMotor, MoveTank,
    OUTPUT_B, OUTPUT_C, OUTPUT_D,
    SpeedPercent
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from drive_calibration import (
    DEFAULT_AXLE_TRACK_MM,
    DEFAULT_MM_PER_UNIT,
    DriveCalibrationValues,
    format_drive_calibration_response,
    load_drive_calibration,
    save_drive_calibration,
    validate_drive_calibration,
)

HOST = '0.0.0.0'
PORT = 5555

# --- Robot physical configuration (tune these to your robot) ---
WHEEL_DIAMETER_MM = 56.0             # Diameter of drive wheels in mm
AXLE_TRACK_MM = DEFAULT_AXLE_TRACK_MM      # Center-to-center distance between drive wheels in mm
MM_PER_UNIT = DEFAULT_MM_PER_UNIT          # Linear travel mm per move/back unit.
DRIVE_CALIBRATION_FILE = Path(__file__).with_name("robot_drive_calibration.json")
_loaded_drive_calibration = load_drive_calibration(
    DRIVE_CALIBRATION_FILE,
    DriveCalibrationValues(AXLE_TRACK_MM, MM_PER_UNIT),
)
AXLE_TRACK_MM = _loaded_drive_calibration.axle_track_mm
MM_PER_UNIT = _loaded_drive_calibration.mm_per_unit

# Pipe motor: degrees of motor rotation per unit of pipe travel
# Tune this based on your pipe mechanism's gear ratio / spool size
PIPE_DEGREES_PER_UNIT = 45.0
COLLECTOR_TRAVEL_UNITS = 5
COLLECTOR_TRAVEL_SPEED = 50
PICKUP_ASSIST_UNITS = 25
PICKUP_ASSIST_SPEED = 40
UNLOAD_FULL_CYCLE_UNITS = 25
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


def current_drive_calibration():
    """Return the active drive calibration constants."""
    return DriveCalibrationValues(
        axle_track_mm=float(AXLE_TRACK_MM),
        mm_per_unit=float(MM_PER_UNIT),
    )


def set_drive_calibration(axle_track_mm, mm_per_unit):
    """Validate, persist, and apply drive calibration constants."""
    global AXLE_TRACK_MM, MM_PER_UNIT
    values = validate_drive_calibration(
        DriveCalibrationValues(float(axle_track_mm), float(mm_per_unit))
    )
    save_drive_calibration(DRIVE_CALIBRATION_FILE, values)
    AXLE_TRACK_MM = values.axle_track_mm
    MM_PER_UNIT = values.mm_per_unit
    return values


# ---------------------------------------------------------------------------
# Motor mounting helpers — single point of negation for reversed motors
# ---------------------------------------------------------------------------

def _apply_wheel_speeds(left_pct, right_pct):
    """Continuous drive — single point of motor-mounting negation."""
    if abs(left_pct) < 1e-6 and abs(right_pct) < 1e-6:
        tank.off()
    else:
        tank.on(left_speed=SpeedPercent(-left_pct), right_speed=SpeedPercent(-right_pct))


def _apply_wheel_speeds_for_degrees(left_pct, right_pct, motor_degrees):
    """Blocking drive — single point of motor-mounting negation."""
    tank.on_for_degrees(
        left_speed=SpeedPercent(-left_pct), right_speed=SpeedPercent(-right_pct),
        degrees=motor_degrees, brake=True, block=True,
    )


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
    _apply_wheel_speeds_for_degrees(spd, spd, motor_deg)
    direction = "forward" if forward else "backward"
    return "ok: moved {} {} units".format(direction, units)


def cmd_turn(parts):
    if len(parts) < 2:
        return "error: turn requires <degrees> [speed]"
    angle = float(parts[1])
    speed = float(parts[2]) if len(parts) > 2 else 30
    motor_deg = turn_angle_to_motor_degrees(angle)

    # Positive angle = CW (right), negative = CCW (left).
    if angle < 0:
        left_spd, right_spd = -speed, speed   # left backward, right forward = CCW
    else:
        left_spd, right_spd = speed, -speed   # left forward, right backward = CW

    _apply_wheel_speeds_for_degrees(left_spd, right_spd, motor_deg)
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


def cmd_lr(parts):
    if len(parts) < 3:
        return "error: LR requires <left_speed_pct> <right_speed_pct>"
    left = float(parts[1])
    right = float(parts[2])
    if not (math.isfinite(left) and math.isfinite(right)):
        return "error: non-finite wheel speed"
    _apply_wheel_speeds(left, right)
    return "ok: wheel speeds {} {}".format(left, right)


def cmd_drivecal(parts):
    if len(parts) < 2:
        return "error: drivecal requires get or set <axle_track_mm> <mm_per_unit>"
    subaction = parts[1].lower()
    if subaction == "get":
        return format_drive_calibration_response(current_drive_calibration())
    if subaction == "set":
        if len(parts) != 4:
            return "error: drivecal set requires <axle_track_mm> <mm_per_unit>"
        values = set_drive_calibration(float(parts[2]), float(parts[3]))
        return format_drive_calibration_response(values)
    return "error: drivecal requires get or set <axle_track_mm> <mm_per_unit>"


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
        elif action == "drivecal":
            return cmd_drivecal(parts)
        elif action == "lr":
            return cmd_lr(parts)
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
    # Newline-framed parsing: a single recv() can carry several commands when the
    # client streams wheel speeds fire-and-forget (e.g. "LR 30 30\nLR 35 35\n").
    # Splitting on '\n' processes each one in order (last wheel command wins)
    # instead of merging them into a single mangled token.
    buffer = ""
    with conn:
        while True:
            try:
                data = conn.recv(1024)
                if not data:
                    break
                buffer += data.decode('utf-8')
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    cmd = line.strip()
                    if not cmd:
                        continue
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
