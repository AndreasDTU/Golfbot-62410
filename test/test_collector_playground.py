import unittest

from tools import collector_playground
from tools.collector_playground import (
    CollectorBelief,
    CollectorPlayground,
    CollectorPlaygroundGui,
    DummyCollectorController,
    build_arg_parser,
    gui_command_text,
)


class FakeCollectorController:
    def __init__(self) -> None:
        self.calls: list[tuple[str, float | None]] = []

    def collector_travel_position(self) -> str:
        self.calls.append(("collector_travel_position", None))
        return "ok travel"

    def pickup_assist(self) -> str:
        self.calls.append(("pickup_assist", None))
        return "ok assist"

    def unload_full_cycle(self) -> str:
        self.calls.append(("unload_full_cycle", None))
        return "ok unload"

    def pipe_up(self, units, speed=None) -> str:
        self.calls.append(("pipe_up", units))
        return "ok up"

    def pipe_down(self, units, speed=None) -> str:
        self.calls.append(("pipe_down", units))
        return "ok down"

    def pipe_stop(self) -> str:
        self.calls.append(("pipe_stop", None))
        return "ok pipe stop"

    def stop(self) -> str:
        self.calls.append(("stop", None))
        return "ok stop"


class CollectorPlaygroundTests(unittest.TestCase):
    def make_playground(self, confirm_unload: bool = False) -> tuple[CollectorPlayground, FakeCollectorController]:
        controller = FakeCollectorController()
        playground = CollectorPlayground(controller, confirm_unload=confirm_unload, max_manual_units=5.0)
        return playground, controller

    def test_travel_maps_to_collector_travel_position(self) -> None:
        playground, controller = self.make_playground()

        playground.execute("travel")

        self.assertEqual(controller.calls, [("collector_travel_position", None)])
        self.assertEqual(playground.state, CollectorBelief.TRAVEL)

    def test_assist_and_pickup_map_to_pickup_assist(self) -> None:
        playground, controller = self.make_playground()

        playground.execute("assist")
        playground.execute("pickup")

        self.assertEqual(controller.calls, [("pickup_assist", None), ("pickup_assist", None)])
        self.assertEqual(playground.state, CollectorBelief.TRAVEL)

    def test_unload_and_dropoff_map_to_unload_full_cycle_when_confirmation_disabled(self) -> None:
        playground, controller = self.make_playground(confirm_unload=False)

        playground.execute("unload")
        playground.execute("dropoff")

        self.assertEqual(controller.calls, [("unload_full_cycle", None), ("unload_full_cycle", None)])
        self.assertEqual(playground.state, CollectorBelief.UNKNOWN)

    def test_invalid_up_down_arguments_are_rejected(self) -> None:
        playground, controller = self.make_playground()

        _, missing = playground.execute("up")
        _, non_numeric = playground.execute("down nope")
        _, too_large = playground.execute("up 6")

        self.assertIn("Usage", missing)
        self.assertIn("numeric", non_numeric)
        self.assertIn("exceed", too_large)
        self.assertEqual(controller.calls, [])

    def test_valid_up_down_are_bounded_and_sent(self) -> None:
        playground, controller = self.make_playground()

        playground.execute("up 3")
        playground.execute("down 2.5")

        self.assertEqual(controller.calls, [("pipe_up", 3.0), ("pipe_down", 2.5)])

    def test_unload_asks_for_confirmation_unless_disabled(self) -> None:
        prompts: list[str] = []
        controller = FakeCollectorController()
        playground = CollectorPlayground(
            controller,
            confirm_unload=True,
            confirm=lambda prompt: prompts.append(prompt) or False,
        )

        _, message = playground.execute("unload")

        self.assertEqual(controller.calls, [])
        self.assertEqual(len(prompts), 1)
        self.assertIn("cancelled", message.lower())

    def test_state_observer_sees_pickup_assist_then_travel(self) -> None:
        controller = FakeCollectorController()
        states: list[CollectorBelief] = []
        playground = CollectorPlayground(controller, confirm_unload=False, on_state_change=states.append)

        playground.execute("assist")

        self.assertEqual(states, [CollectorBelief.PICKUP_ASSIST, CollectorBelief.TRAVEL])

    def test_full_unload_does_not_happen_on_startup(self) -> None:
        playground, controller = self.make_playground(confirm_unload=False)

        self.assertEqual(playground.state, CollectorBelief.UNKNOWN)
        self.assertEqual(controller.calls, [])

    def test_gui_button_commands_use_terminal_command_text(self) -> None:
        self.assertEqual(gui_command_text("travel", "3"), "travel")
        self.assertEqual(gui_command_text("assist", "3"), "assist")
        self.assertEqual(gui_command_text("unload", "3"), "unload")
        self.assertEqual(gui_command_text("up", " 2.5 "), "up 2.5")
        self.assertEqual(gui_command_text("down", "bad"), "down bad")

    def test_dummy_controller_records_commands_without_network(self) -> None:
        controller = DummyCollectorController()
        playground = CollectorPlayground(controller, confirm_unload=False, max_manual_units=5.0)

        playground.execute("travel")
        playground.execute("up 2")
        playground.execute("stop")

        self.assertEqual(controller.commands, ["collector_travel_position", "pipe up 2.0", "pipe stop"])

    def test_dummy_cli_flag_parses_without_host_override(self) -> None:
        args = build_arg_parser().parse_args(["--dummy"])

        self.assertTrue(args.dummy)
        self.assertFalse(args.cli)

    def test_dummy_gui_renders_without_robot_connection(self) -> None:
        args = build_arg_parser().parse_args(["--dummy"])
        gui = CollectorPlaygroundGui(args)

        image = gui.render()

        self.assertEqual(image.shape, (680, 1000, 3))
        self.assertTrue(gui.connected)
        self.assertIn("Dummy mode", gui.connection_label)
        gui.close()

    def test_module_does_not_import_autonomous_stack(self) -> None:
        forbidden = (
            "VisionPipeline",
            "RoutePlanningFacade",
            "TopDownDetectorApp",
            "UdpWheelDispatcher",
        )
        for name in forbidden:
            self.assertFalse(hasattr(collector_playground, name), name)


if __name__ == "__main__":
    unittest.main()
