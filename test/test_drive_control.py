import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from pathfinding.models import HybridPose, RoutePlan, RouteTrackingError
from robot.control import WheelCommandController, robot_body_edge_clearance_cm, route_goal_pose
from robot.models import DriveControlState, DriveRuntime, RobotGeometry, RobotPose
from tools.topdown_object_detector import CollectorPositionState, PickupExecutionState, TopdownDetectorApp
from vision.config import DriveConfig, FieldConfig
from vision.debug import DebugRenderer


class FakeDispatcher:
    def __init__(self) -> None:
        self.commands: list[tuple[float, float, bool]] = []
        self.last_error = ""

    def send_wheel_speeds(self, left_pct: float, right_pct: float, force: bool = False) -> bool:
        self.commands.append((left_pct, right_pct, force))
        return True


class DriveControlTests(unittest.TestCase):
    def test_body_edge_clearance_uses_main_body_footprint(self) -> None:
        pose = RobotPose(x_cm=20.0, y_cm=20.0, heading_rad=0.0, tube_x_cm=35.0, tube_y_cm=20.0)
        geometry = RobotGeometry(width_cm=20.0, front_cm=8.0, rear_cm=10.0, tube_forward_cm=18.0, tube_right_cm=0.0)

        clearance = robot_body_edge_clearance_cm(pose, geometry, FieldConfig(width_cm=167.0, height_cm=121.5))

        self.assertAlmostEqual(clearance, 10.0)

    def test_edge_scaling_slows_forward_speed_and_increases_correction(self) -> None:
        config = DriveConfig()
        error = RouteTrackingError(
            xte_cm=2.0,
            signed_xte_cm=2.0,
            heading_error_rad=0.1,
            closest_point_cm=(0.0, 0.0),
            segment_heading_rad=0.0,
            segment_index=0,
        )

        open_field = WheelCommandController(config).compute(error, distance_to_goal_cm=50.0, edge_clearance_cm=30.0)
        near_wall = WheelCommandController(config).compute(error, distance_to_goal_cm=50.0, edge_clearance_cm=0.0)

        self.assertLess((near_wall.left_pct + near_wall.right_pct) * 0.5, (open_field.left_pct + open_field.right_pct) * 0.5)
        self.assertGreater(abs(near_wall.right_pct - near_wall.left_pct), abs(open_field.right_pct - open_field.left_pct))

    def test_route_goal_pose_uses_next_pickup_checkpoint(self) -> None:
        route = [HybridPose(0.0, 0.0, 0.0), HybridPose(10.0, 0.0, 0.0), HybridPose(20.0, 0.0, 0.0)]
        pickup = [HybridPose(10.0, 0.0, 0.0)]
        error = RouteTrackingError(0.0, 0.0, 0.0, (0.0, 0.0), 0.0, 0)

        self.assertEqual(route_goal_pose(route, error, pickup, include_final=False), route[1])

    def test_route_heatmap_breaks_at_near_zone_boundary_for_each_pickup(self) -> None:
        renderer = DebugRenderer(drive_config=DriveConfig(near_zone_cm=15.0))
        geometry = RobotGeometry(width_cm=20.0, front_cm=8.0, rear_cm=10.0, tube_forward_cm=0.0, tube_right_cm=0.0)
        route = [
            HybridPose(0.0, 0.0, 0.0),
            HybridPose(20.0, 0.0, 0.0),
            HybridPose(40.0, 0.0, 0.0),
            HybridPose(60.0, 0.0, 0.0),
            HybridPose(80.0, 0.0, 0.0),
        ]
        pickups = [route[2], route[4]]

        breaks = renderer.near_zone_visual_breaks(route, pickups, geometry)

        self.assertEqual(len(breaks), 2)
        self.assertEqual([visual_break.checkpoint_index for visual_break in breaks], [2, 4])
        self.assertAlmostEqual(breaks[0].boundary_pose.x_cm, 25.0)
        self.assertAlmostEqual(breaks[1].boundary_pose.x_cm, 65.0)
        self.assertEqual(breaks[0].final_pickup_pose, route[2])
        self.assertEqual(breaks[1].final_pickup_pose, route[4])

    def test_ball_count_initializes_after_stable_detection_window(self) -> None:
        app = TopdownDetectorApp()
        result = SimpleNamespace(white_balls=[object(), object()], orange_balls=[object()])

        for index in range(15):
            app.update_ball_count_reconciliation(result, now_s=index * 0.06)

        self.assertEqual(app.runtime.initial_total_balls, 3)
        self.assertEqual(app.runtime.stable_visible_balls, 3)
        self.assertEqual(app.runtime.balls_collected, 0)

    def test_ball_count_debounce_rejects_flicker_before_reconciliation(self) -> None:
        app = TopdownDetectorApp()
        app.runtime.initial_total_balls = 3
        app.runtime.balls_collected = 1

        for index, count in enumerate([2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2]):
            result = SimpleNamespace(white_balls=[object()] * count, orange_balls=[])
            app.update_ball_count_reconciliation(result, now_s=index * 0.06)

        self.assertIsNone(app.runtime.stable_visible_balls)
        self.assertEqual(app.runtime.balls_collected, 1)

    def test_ball_count_reconciliation_decrements_missed_pickup(self) -> None:
        app = TopdownDetectorApp()
        app.runtime.initial_total_balls = 3
        app.runtime.balls_collected = 1
        result = SimpleNamespace(white_balls=[object(), object(), object()], orange_balls=[])

        for index in range(15):
            app.update_ball_count_reconciliation(result, now_s=index * 0.06)

        self.assertEqual(app.runtime.stable_visible_balls, 3)
        self.assertEqual(app.runtime.balls_collected, 0)

    def test_step_drive_release_key_unpauses_without_sending_motion(self) -> None:
        app = TopdownDetectorApp()
        dispatcher = FakeDispatcher()
        drive_runtime = DriveRuntime(enabled=True, dispatcher=dispatcher)
        app.runtime.step_mode_enabled = True
        app.runtime.step_drive_paused = True
        app.runtime.initial_total_balls = 1
        app.runtime.robot_pose = RobotPose(0.0, 0.0, 0.0, 0.0, 0.0)
        app.runtime.route_plan = RoutePlan(
            points=[HybridPose(0.0, 0.0, 0.0), HybridPose(10.0, 0.0, 0.0)],
            active_target=None,
            pickup_poses=[],
        )

        should_quit = app.handle_key(ord("n"), drive_runtime)

        self.assertFalse(should_quit)
        self.assertFalse(app.runtime.step_drive_paused)
        self.assertEqual(dispatcher.commands, [])
        self.assertEqual(drive_runtime.last_message, "step released")

    def test_step_drive_release_key_ignores_stale_press_before_route_ready(self) -> None:
        app = TopdownDetectorApp()
        dispatcher = FakeDispatcher()
        drive_runtime = DriveRuntime(enabled=True, dispatcher=dispatcher)
        app.runtime.step_mode_enabled = True
        app.runtime.step_drive_paused = True
        app.runtime.initial_total_balls = 1
        app.runtime.robot_pose = RobotPose(0.0, 0.0, 0.0, 0.0, 0.0)

        should_quit = app.handle_key(ord("n"), drive_runtime)

        self.assertFalse(should_quit)
        self.assertTrue(app.runtime.step_drive_paused)
        self.assertEqual(dispatcher.commands, [])
        self.assertEqual(drive_runtime.last_message, "step waiting for route")

    def test_step_drive_pause_after_target_sends_zero_speed(self) -> None:
        app = TopdownDetectorApp()
        dispatcher = FakeDispatcher()
        drive_runtime = DriveRuntime(enabled=True, dispatcher=dispatcher)
        app.runtime.step_mode_enabled = True
        app.runtime.step_drive_paused = False

        app.pause_step_drive_after_target(drive_runtime)

        self.assertTrue(app.runtime.step_drive_paused)
        self.assertEqual(drive_runtime.state, DriveControlState.STOPPED)
        self.assertEqual(drive_runtime.last_message, "step target complete; press n")
        self.assertEqual(dispatcher.commands[-1][:2], (0.0, 0.0))

    def test_near_zone_handoff_stops_udp_then_turns_and_runs_tcp_move(self) -> None:
        app = TopdownDetectorApp()
        dispatcher = FakeDispatcher()
        drive_runtime = DriveRuntime(enabled=True, dispatcher=dispatcher)
        app.runtime.initial_total_balls = 1
        app.runtime.robot_pose = RobotPose(
            x_cm=6.0,
            y_cm=0.0,
            heading_rad=math.radians(10.0),
            tube_x_cm=0.0,
            tube_y_cm=0.0,
        )
        app.runtime.route_plan = RoutePlan(
            points=[HybridPose(0.0, 0.0, 0.0), HybridPose(10.0, 0.0, 0.0)],
            active_target=None,
            pickup_poses=[HybridPose(10.0, 0.0, 0.0)],
        )

        events = []

        def record_stop(left_pct: float, right_pct: float, force: bool = False) -> bool:
            dispatcher.commands.append((left_pct, right_pct, force))
            events.append(("udp", left_pct, None))
            return True

        dispatcher.send_wheel_speeds = record_stop

        class FakeRobotController:
            def __init__(self, robot_ip: str) -> None:
                self.robot_ip = robot_ip

            def turn(self, degrees: float, speed_percent: int) -> str:
                events.append(("turn", degrees, speed_percent))
                return "OK"

            def move(self, distance: float, speed_percent: int) -> str:
                events.append(("move", distance, speed_percent))
                return "OK"

        with patch("tools.topdown_object_detector.RobotController", FakeRobotController):
            owns_control = app.update_pickup_state(drive_runtime, now_s=1.0)

        self.assertTrue(owns_control)
        self.assertEqual(dispatcher.commands[-1][:2], (0.0, 0.0))
        self.assertEqual(events[0], ("udp", 0.0, None))
        self.assertEqual(events[1], ("turn", 10.0, int(round(app.config.drive.near_zone_turn_speed_pct))))
        self.assertEqual(events[2], ("move", 4.0, int(round(app.config.drive.near_zone_move_speed_pct))))
        self.assertEqual(app.runtime.balls_collected, 1)
        self.assertEqual(app.runtime.pickup_state, PickupExecutionState.PICKUP_ASSIST)

    def test_autonomous_collection_runs_pickup_assist_not_full_unload(self) -> None:
        app = TopdownDetectorApp()
        dispatcher = FakeDispatcher()
        drive_runtime = DriveRuntime(enabled=True, dispatcher=dispatcher)
        app.runtime.pickup_state = PickupExecutionState.PICKUP_ASSIST
        events = []

        class FakeRobotController:
            def __init__(self, robot_ip: str) -> None:
                self.robot_ip = robot_ip

            def pickup_assist(self) -> str:
                events.append("pickup_assist")
                return "OK"

            def unload_full_cycle(self) -> str:
                events.append("unload_full_cycle")
                return "OK"

        with patch("tools.topdown_object_detector.RobotController", FakeRobotController):
            owns_control = app.update_pickup_state(drive_runtime, now_s=1.0)
            assert app.pickup_thread is not None
            app.pickup_thread.join(timeout=1.0)
            if app.runtime.pickup_state == PickupExecutionState.PICKUP_ASSIST:
                app.update_pickup_state(drive_runtime, now_s=1.1)

        self.assertTrue(owns_control)
        self.assertEqual(events, ["pickup_assist"])
        self.assertEqual(app.runtime.collector_state, CollectorPositionState.TRAVEL)
        self.assertEqual(app.runtime.pickup_state, PickupExecutionState.REPLAN)
        self.assertEqual(drive_runtime.last_message, "pickup assist")

    def test_orange_autonomous_collection_runs_same_pickup_assist(self) -> None:
        app = TopdownDetectorApp()
        dispatcher = FakeDispatcher()
        drive_runtime = DriveRuntime(enabled=True, dispatcher=dispatcher)
        app.runtime.pickup_state = PickupExecutionState.PICKUP_ASSIST
        app.runtime.route_plan = RoutePlan(
            points=[HybridPose(0.0, 0.0, 0.0)],
            active_target=SimpleNamespace(label="orange"),
            pickup_poses=[HybridPose(0.0, 0.0, 0.0)],
        )
        events = []

        class FakeRobotController:
            def __init__(self, robot_ip: str) -> None:
                self.robot_ip = robot_ip

            def pickup_assist(self) -> str:
                events.append("pickup_assist")
                return "OK"

            def unload_full_cycle(self) -> str:
                events.append("unload_full_cycle")
                return "OK"

        with patch("tools.topdown_object_detector.RobotController", FakeRobotController):
            owns_control = app.update_pickup_state(drive_runtime, now_s=1.0)
            assert app.pickup_thread is not None
            app.pickup_thread.join(timeout=1.0)
            if app.runtime.pickup_state == PickupExecutionState.PICKUP_ASSIST:
                app.update_pickup_state(drive_runtime, now_s=1.1)

        self.assertTrue(owns_control)
        self.assertEqual(events, ["pickup_assist"])
        self.assertEqual(app.runtime.collector_state, CollectorPositionState.TRAVEL)

    def test_drive_start_requires_collector_travel_position_before_route_following(self) -> None:
        app = TopdownDetectorApp()
        dispatcher = FakeDispatcher()
        drive_runtime = DriveRuntime(enabled=True, dispatcher=dispatcher)
        events = []

        class FakeRobotController:
            def __init__(self, robot_ip: str) -> None:
                self.robot_ip = robot_ip

            def collector_travel_position(self) -> str:
                events.append("collector_travel_position")
                return "OK"

        with patch("tools.topdown_object_detector.RobotController", FakeRobotController):
            owns_control = app.ensure_collector_travel_position(drive_runtime)

        self.assertTrue(owns_control)
        self.assertEqual(events, ["collector_travel_position"])
        self.assertEqual(app.runtime.collector_state, CollectorPositionState.TRAVEL)
        self.assertEqual(dispatcher.commands[-1][:2], (0.0, 0.0))

    def test_route_following_not_blocked_after_collector_travel_confirmed(self) -> None:
        app = TopdownDetectorApp()
        drive_runtime = DriveRuntime(enabled=True, dispatcher=FakeDispatcher())
        app.runtime.collector_state = CollectorPositionState.TRAVEL

        owns_control = app.ensure_collector_travel_position(drive_runtime)

        self.assertFalse(owns_control)


if __name__ == "__main__":
    unittest.main()
