import unittest

from robot.controller import RobotController


class CollectionActuatorCommandTests(unittest.TestCase):
    def controller_with_recorder(self) -> tuple[RobotController, list[str]]:
        controller = RobotController.__new__(RobotController)
        sent: list[str] = []
        controller._send = sent.append
        return controller, sent

    def test_pickup_assist_sends_collection_only_command(self) -> None:
        controller, sent = self.controller_with_recorder()

        controller.pickup_assist()

        self.assertEqual(sent, ["pickup_assist"])

    def test_unload_full_cycle_sends_full_unload_command(self) -> None:
        controller, sent = self.controller_with_recorder()

        controller.unload_full_cycle()

        self.assertEqual(sent, ["unload_full_cycle"])

    def test_compatibility_pickup_alias_does_not_send_full_unload(self) -> None:
        controller, sent = self.controller_with_recorder()

        controller.pickup()

        self.assertEqual(sent, ["pickup_assist"])

    def test_compatibility_dropoff_alias_sends_full_unload(self) -> None:
        controller, sent = self.controller_with_recorder()

        controller.dropoff()

        self.assertEqual(sent, ["unload_full_cycle"])


if __name__ == "__main__":
    unittest.main()
