import atexit
import importlib.util
import sys
import tempfile
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
            REPO_ROOT / "control" / "robot" / "robot_server.py",
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        test_path = Path(tempfile.gettempdir()) / f"golfbot_drivecal_test_{id(module)}.json"
        test_path.unlink(missing_ok=True)
        atexit.register(lambda path=test_path: path.unlink(missing_ok=True))
        module.DRIVE_CALIBRATION_FILE = test_path
        module.AXLE_TRACK_MM = module.DEFAULT_AXLE_TRACK_MM
        module.MM_PER_UNIT = module.DEFAULT_MM_PER_UNIT
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

    def test_drivecal_get_returns_loaded_constants(self) -> None:
        robot_server = load_robot_server_with_fake_ev3()

        response = robot_server.handle_command("drivecal get")

        self.assertIn("ok: drivecal", response)
        self.assertIn("axle_track_mm", response)
        self.assertIn("mm_per_unit", response)

    def test_drivecal_set_persists_and_updates_conversion_helpers(self) -> None:
        robot_server = load_robot_server_with_fake_ev3()
        with tempfile.TemporaryDirectory() as tmp:
            robot_server.DRIVE_CALIBRATION_FILE = Path(tmp) / "robot_drive_calibration.json"

            response = robot_server.handle_command("drivecal set 300.0 12.5")

            self.assertEqual(response, "ok: drivecal axle_track_mm 300.000000 mm_per_unit 12.500000")
            self.assertTrue(robot_server.DRIVE_CALIBRATION_FILE.exists())
            wheel_circumference_mm = 3.14159 * robot_server.WHEEL_DIAMETER_MM
            expected_turn_degrees = (3.14159 * 300.0 / wheel_circumference_mm) * 360.0
            expected_move_degrees = (10.0 * 12.5 / wheel_circumference_mm) * 360.0
            self.assertAlmostEqual(robot_server.turn_angle_to_motor_degrees(360.0), expected_turn_degrees)
            self.assertAlmostEqual(robot_server.units_to_degrees(10.0), expected_move_degrees)

    def test_drivecal_rejects_invalid_values(self) -> None:
        robot_server = load_robot_server_with_fake_ev3()
        original = robot_server.current_drive_calibration()

        response = robot_server.handle_command("drivecal set nan 12.5")

        self.assertTrue(response.startswith("error:"))
        self.assertEqual(robot_server.current_drive_calibration(), original)


if __name__ == "__main__":
    unittest.main()
