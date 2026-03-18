#!/usr/bin/env python3
"""
EV3 Calibration Tool
Run this on your laptop while robot_server.py is running on the EV3.

Usage:
    python3 calibrate.py <EV3_IP>

Workflow (drive):
    1. Type: move 10
    2. Measure how far the robot actually moved (in cm)
    3. Type: calibrate move 10 9   (asked, actual)

Workflow (turn):
    1. Type: turn 360
    2. Measure how many degrees it actually turned (easiest: mark the floor)
    3. Type: calibrate turn 360 320   (asked, actual)

Tip: use 'turn 360' for turn calibration — a full rotation is easy to measure.
"""

import socket
import sys

# Drive calibration: 1 unit = 10mm = 1cm on the server by default
MM_PER_UNIT = 10.0

# Turn calibration: multiplier on top of the server's turn calculation
# 1.0 = no correction. >1.0 = robot underturns, <1.0 = robot overturns
TURN_SCALE = 1.0

def connect(host, port=5555):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(15.0)
    s.connect((host, port))
    return s

def send(sock, cmd):
    sock.sendall((cmd + '\n').encode('utf-8'))
    return sock.recv(1024).decode('utf-8').strip()

def scale_cmd(raw_cmd):
    """
    Intercept move/back/turn commands and apply current calibration scale.
    Returns (scaled_cmd, was_changed).
    """
    parts = raw_cmd.strip().split()
    if not parts:
        return raw_cmd, False

    verb = parts[0].lower()

    if verb in ('move', 'back') and len(parts) >= 2:
        try:
            units = float(parts[1])
            scaled = units * (MM_PER_UNIT / 10.0)
            parts[1] = str(round(scaled, 4))
            return ' '.join(parts), scaled != units
        except ValueError:
            pass

    if verb == 'turn' and len(parts) >= 2:
        try:
            degrees = float(parts[1])
            scaled = degrees * TURN_SCALE
            parts[1] = str(round(scaled, 4))
            return ' '.join(parts), scaled != degrees
        except ValueError:
            pass

    return raw_cmd, False

def print_help():
    print("""
Commands:
  move <cm>                     Drive forward N cm
  back <cm>                     Drive backward N cm
  turn <degrees>                Turn in place (+ right, - left)
  pipe up <n>                   Raise pipe
  pipe down <n>                 Lower pipe
  stop                          Stop motors
  ping                          Check connection

  calibrate move <asked> <actual>   Adjust drive scale
    e.g. 'calibrate move 10 9' if you asked 10cm but got 9cm

  calibrate turn <asked> <actual>   Adjust turn scale
    e.g. 'calibrate turn 360 310' if you asked 360deg but got 310deg
    Tip: use 360 as your test -- easy to see a full rotation on the floor

  scale                         Show current calibration values
  help                          Show this message
  quit                          Print final values and exit
""")

def print_scale():
    print("  MM_PER_UNIT  = {:.4f}  (drive,  server default 10.0)".format(MM_PER_UNIT))
    print("  TURN_SCALE   = {:.4f}  (turn multiplier, 1.0 = no correction)".format(TURN_SCALE))
    print("")
    print("  To apply in robot_server.py:")
    print("    MM_PER_UNIT       = {:.4f}".format(MM_PER_UNIT))
    print("    AXLE_TRACK_MM    *= {:.4f}   # multiply your current value by this".format(TURN_SCALE))

def main():
    global MM_PER_UNIT, TURN_SCALE

    if len(sys.argv) < 2:
        print("Usage: python3 calibrate.py <EV3_IP>")
        sys.exit(1)

    host = sys.argv[1]
    print("Connecting to {}...".format(host))
    sock = connect(host)
    print("Connected. Type 'help' for commands.\n")
    print_scale()

    while True:
        try:
            raw = input("ev3> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not raw:
            continue

        parts = raw.split()
        cmd = parts[0].lower()

        # --- Local commands ---

        if cmd == 'quit':
            print("\nFinal calibration values:")
            print_scale()
            break

        elif cmd == 'help':
            print_help()

        elif cmd == 'scale':
            print_scale()

        elif cmd == 'calibrate':
            # calibrate move <asked> <actual>
            # calibrate turn <asked> <actual>
            if len(parts) != 4:
                print("Usage: calibrate move <asked> <actual>")
                print("       calibrate turn <asked> <actual>")
                continue

            kind = parts[1].lower()
            if kind not in ('move', 'turn'):
                print("Specify 'move' or 'turn', e.g: calibrate move 10 9")
                continue

            try:
                asked  = float(parts[2])
                actual = float(parts[3])
                if actual == 0:
                    print("Actual value can't be zero.")
                    continue

                ratio = asked / actual

                if kind == 'move':
                    MM_PER_UNIT = MM_PER_UNIT * ratio
                    print("Drive adjusted: MM_PER_UNIT = {:.4f}".format(MM_PER_UNIT))
                else:
                    TURN_SCALE = TURN_SCALE * ratio
                    print("Turn adjusted:  TURN_SCALE  = {:.4f}".format(TURN_SCALE))
                    print("  (In robot_server.py, multiply AXLE_TRACK_MM by {:.4f})".format(TURN_SCALE))

            except ValueError:
                print("Both values must be numbers.")

        # --- Robot commands (forwarded over socket) ---

        else:
            scaled_cmd, was_scaled = scale_cmd(raw)
            if was_scaled:
                print("(sending: {})".format(scaled_cmd))
            response = send(sock, scaled_cmd)
            print(response)

    sock.close()

if __name__ == '__main__':
    main()
