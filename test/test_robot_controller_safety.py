import socket
import unittest
from typing import Optional
from unittest.mock import patch

from robot.controller import RobotController


class FakeSocket:
    def __init__(
        self,
        *,
        recv_error: Optional[OSError] = None,
        recv_payload: bytes = b"OK\n",
    ) -> None:
        self.recv_error = recv_error
        self.recv_payload = recv_payload
        self.sent_payloads: list[bytes] = []
        self.connected_to: tuple[str, int] | None = None
        self.timeout: float | None = None
        self.closed = False

    def settimeout(self, timeout: float) -> None:
        self.timeout = timeout

    def connect(self, address: tuple[str, int]) -> None:
        self.connected_to = address

    def sendall(self, payload: bytes) -> None:
        self.sent_payloads.append(payload)

    def recv(self, _size: int) -> bytes:
        if self.recv_error is not None:
            raise self.recv_error
        return self.recv_payload

    def close(self) -> None:
        self.closed = True


class RobotControllerSafetyTests(unittest.TestCase):
    def test_non_idempotent_command_is_not_replayed_after_ack_failure(self) -> None:
        fake_socket = FakeSocket(recv_error=socket.timeout("timed out"))

        with patch("robot.controller.socket.socket", return_value=fake_socket):
            controller = RobotController("robot.local", timeout=0.1)
            with self.assertRaises(RuntimeError):
                controller.move(10, 25)

        self.assertEqual(fake_socket.sent_payloads, [b"move 10 25\n"])

    def test_command_failure_surfaces_context(self) -> None:
        fake_socket = FakeSocket(recv_error=socket.timeout("timed out"))

        with patch("robot.controller.socket.socket", return_value=fake_socket):
            controller = RobotController("robot.local", timeout=0.1)
            with self.assertRaisesRegex(RuntimeError, "EV3 command failed"):
                controller.unload_full_cycle()

    def test_wheel_speed_command_uses_tcp_lr_protocol(self) -> None:
        fake_socket = FakeSocket()

        with patch("robot.controller.socket.socket", return_value=fake_socket):
            controller = RobotController("robot.local", timeout=0.1)
            response = controller.send_wheel_speeds(12.5, -7.0)

        self.assertEqual(response, "OK")
        self.assertEqual(fake_socket.sent_payloads, [b"LR 12.5 -7.0\n"])

    def test_drive_calibration_get_parses_drivecal_response(self) -> None:
        fake_socket = FakeSocket(
            recv_payload=b"ok: drivecal axle_track_mm 300.000000 mm_per_unit 12.500000\n"
        )

        with patch("robot.controller.socket.socket", return_value=fake_socket):
            controller = RobotController("robot.local", timeout=0.1)
            values = controller.get_drive_calibration()

        self.assertEqual(values.axle_track_mm, 300.0)
        self.assertEqual(values.mm_per_unit, 12.5)
        self.assertEqual(fake_socket.sent_payloads, [b"drivecal get\n"])

    def test_drive_calibration_set_sends_drivecal_protocol(self) -> None:
        fake_socket = FakeSocket(
            recv_payload=b"ok: drivecal axle_track_mm 300.000000 mm_per_unit 12.500000\n"
        )

        with patch("robot.controller.socket.socket", return_value=fake_socket):
            controller = RobotController("robot.local", timeout=0.1)
            values = controller.set_drive_calibration(300.0, 12.5)

        self.assertEqual(values.axle_track_mm, 300.0)
        self.assertEqual(values.mm_per_unit, 12.5)
        self.assertEqual(fake_socket.sent_payloads, [b"drivecal set 300.000000 12.500000\n"])


if __name__ == "__main__":
    unittest.main()
