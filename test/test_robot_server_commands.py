import importlib.util
import sys
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class FakeTank:
    def __init__(self, *_args, **_kwargs) -> None:
        self.calls: list[dict[str, object]] = []
        self.on_calls: list[dict[str, object]] = []
        self.off_calls = 0

    def on_for_degrees(self, **kwargs) -> None:
        self.calls.append(kwargs)

    def on(self, **kwargs) -> None:
        self.on_calls.append(kwargs)

    def off(self) -> None:
        self.off_calls += 1


class FakeMotor:
    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def on_for_degrees(self, **_kwargs) -> None:
        pass

    def off(self) -> None:
        pass


def load_robot_server_with_fake_ev3():
    ev3dev2 = types.ModuleType("ev3dev2")
    motor = types.ModuleType("ev3dev2.motor")
    motor.LargeMotor = FakeMotor
    motor.MoveTank = FakeTank
    motor.OUTPUT_B = "outB"
    motor.OUTPUT_C = "outC"
    motor.OUTPUT_D = "outD"
    motor.SpeedPercent = lambda value: value
    previous = {name: sys.modules.get(name) for name in ("ev3dev2", "ev3dev2.motor")}
    sys.modules["ev3dev2"] = ev3dev2
    sys.modules["ev3dev2.motor"] = motor
    try:
        spec = importlib.util.spec_from_file_location(
            "robot_server_under_test",
            REPO_ROOT / "robot" / "robot_server.py",
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


class RobotServerCommandTests(unittest.TestCase):
    def test_move_and_turn_ack_only_after_blocking_motor_completion(self) -> None:
        robot_server = load_robot_server_with_fake_ev3()

        robot_server.cmd_move(["move", "5", "30"])
        robot_server.cmd_turn(["turn", "15", "25"])

        self.assertEqual(len(robot_server.tank.calls), 2)
        self.assertTrue(robot_server.tank.calls[0]["block"])
        self.assertTrue(robot_server.tank.calls[1]["block"])

    def test_lr_tcp_command_sets_continuous_wheel_speeds(self) -> None:
        robot_server = load_robot_server_with_fake_ev3()

        response = robot_server.handle_command("LR 12.5 -7.0")

        self.assertTrue(response.startswith("ok: wheel speeds"))
        self.assertEqual(robot_server.tank.on_calls, [{"left_speed": -12.5, "right_speed": 7.0}])

    def test_lr_tcp_zero_command_stops_drive_motors(self) -> None:
        robot_server = load_robot_server_with_fake_ev3()

        response = robot_server.handle_command("LR 0 0")

        self.assertEqual(response, "ok: wheel speeds 0.0 0.0")
        self.assertEqual(robot_server.tank.off_calls, 1)


if __name__ == "__main__":
    unittest.main()
