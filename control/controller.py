import socket

from control.tools.drive_calibration import DriveCalibrationValues, parse_drive_calibration_response

class RobotController:

    def __init__(self, robot_ip = "ev3dev", port=5555, timeout=15.0, connect_retries=1):
        self.host = robot_ip
        self.port = port
        self.timeout = timeout
        self.connect_retries = max(0, int(connect_retries))
        self.sock = self._connect()

    def _connect(self):
        last_error = None
        for _attempt in range(self.connect_retries + 1):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)
            try:
                sock.connect((self.host, self.port))
                self.sock = sock
                return sock
            except OSError as exc:
                last_error = exc
                sock.close()
        raise RuntimeError(f"Could not connect to EV3 controller at {self.host}:{self.port}: {last_error}") from last_error

    def _send(self, cmd):
        payload = (cmd + '\n').encode('utf-8')
        try:
            self.sock.sendall(payload)
            return self.sock.recv(1024).decode('utf-8').strip()
        except OSError as exc:
            raise RuntimeError(f"EV3 command failed after send attempt ({cmd!r}): {exc}") from exc

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass

    def send_wheel_speeds(self, left_pct, right_pct):
        return self._send(f"LR {left_pct} {right_pct}")
    
    def move(self, distance, speedPercent = 100):
        return self._send(f"move {distance} {speedPercent}")

    def back(self, distance, speedPercent = 100):
        return self._send(f"back {distance} {speedPercent}")

    def turn(self, degrees, speedPercent = 100):
        return self._send(f"turn {degrees} {speedPercent}")

    def get_drive_calibration(self):
        """Read the active EV3 drive calibration constants."""
        return parse_drive_calibration_response(self._send("drivecal get"))

    def set_drive_calibration(self, axle_track_mm, mm_per_unit):
        """Persist and apply EV3 drive calibration constants."""
        values = DriveCalibrationValues(float(axle_track_mm), float(mm_per_unit))
        response = self._send(f"drivecal set {values.axle_track_mm:.6f} {values.mm_per_unit:.6f}")
        return parse_drive_calibration_response(response)

    def collector_travel_position(self):
        """Raise or keep the collector in its safe driving position."""
        return self._send("collector_travel_position")

    def pickup_assist(self):
        """Run the small collection-only pipe jiggle used during ball pickup."""
        return self._send("pickup_assist")

    def unload_full_cycle(self):
        """Run the full pipe cycle used only when unloading at the goal."""
        return self._send("unload_full_cycle")

    def pipe_up(self, units, speed=None):
        """Raise the pipe by an open-loop manual amount."""
        cmd = f"pipe up {units}" if speed is None else f"pipe up {units} {speed}"
        return self._send(cmd)

    def pipe_down(self, units, speed=None):
        """Lower the pipe by an open-loop manual amount."""
        cmd = f"pipe down {units}" if speed is None else f"pipe down {units} {speed}"
        return self._send(cmd)

    def pipe_stop(self):
        """Stop only the pipe motor if the EV3 server supports it."""
        return self._send("pipe stop")

    def pickup(self):
        """Compatibility alias for the collection assist command."""
        return self.pickup_assist()

    def dropoff(self):
        """Compatibility alias for the full unload command."""
        return self.unload_full_cycle()

    def stop(self):
        return self._send("stop")
