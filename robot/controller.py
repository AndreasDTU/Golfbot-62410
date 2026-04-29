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
        except:
            self._connect()
            time.sleep(0.5)
            self._send(cmd)
    
    def move(self, distance, speedPercent = 100):
        self._send(f"move {distance} {speedPercent}")

    def turn(self, degrees, speedPercent = 100):
        self._send(f"turn {degrees} {speedPercent}")

    def pickup(self):
        self._send("pickup")

    def dropoff(self):
        self._send("dropoff")

    def stop(self):
        self._send("stop")