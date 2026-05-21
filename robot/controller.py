import socket
import time

class RobotController:

    def __init__(self, robot_ip = "ev3dev"):
        self.host = robot_ip
        self.sock = self._connect()

    def _connect(self, port=5555):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(15.0)
        self.sock.connect((self.host, port))
        return self.sock

    def _send(self, cmd):
        try:
            self.sock.sendall((cmd + '\n').encode('utf-8'))
            return self.sock.recv(1024).decode('utf-8').strip()
        except OSError:
            self._connect()
            time.sleep(0.5)
            return self._send(cmd)
    
    def move(self, distance, speedPercent = 100):
        return self._send(f"move {distance} {speedPercent}")

    def turn(self, degrees, speedPercent = 100):
        return self._send(f"turn {degrees} {speedPercent}")

    def pickup_assist(self):
        """Run the small collection-only pipe jiggle used during ball pickup."""
        return self._send("pickup_assist")

    def unload_full_cycle(self):
        """Run the full pipe cycle used only when unloading at the goal."""
        return self._send("unload_full_cycle")

    def pickup(self):
        """Compatibility alias for the collection assist command."""
        return self.pickup_assist()

    def dropoff(self):
        """Compatibility alias for the full unload command."""
        return self.unload_full_cycle()

    def stop(self):
        return self._send("stop")
