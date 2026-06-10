import math
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from pathfinding.models import HybridPose, RoutePlan, RouteTrackingError
from robot.control import WheelCommandController, robot_body_edge_clearance_cm, route_goal_pose
from robot.io import TcpWheelDispatcher
from robot.models import DriveControlState, DriveRuntime, RobotGeometry, RobotPose
from tools.topdown_object_detector import (
    CollectorPositionState,
    PickupExecutionState,
    RoutePlanningRequest,
    RoutePlanningResult,
    TopdownDetectorApp,
    UnloadExecutionState,
)
from vision.config import DriveConfig, FieldConfig
from vision.debug import DebugRenderer
from vision.models import SmoothedBallCoordinate


class FakeDispatcher:
    def __init__(self) -> None:
        self.commands: list[tuple[float, float, bool]] = []
        self.last_error = ""

    def send_wheel_speeds(self, left_pct: float, right_pct: float, force: bool = False) -> bool:
        self.commands.append((left_pct, right_pct, force))
        return True


class FakeTcpController:
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.commands: list[str] = []
        self.closed = False

    def _send(self, command: str) -> str:
        self.commands.append(command)
        return "ok"

    def close(self) -> None:
        self.closed = True


class DriveControlTests(unittest.TestCase):
    def test_tcp_wheel_dispatcher_sends_lr_commands_and_deadbands_repeats(self) -> None:
        controllers: list[FakeTcpController] = []

        def make_controller(host: str, port: int) -> FakeTcpController:
            controller = FakeTcpController(host, port)
            controllers.append(controller)
            return controller

        now = [10.0]
        with patch("robot.io.RobotController", make_controller):
            dispatcher = TcpWheelDispatcher(
                host="robot.local",
                port=5555,
                max_speed_pct=50.0,
                min_send_interval_s=0.1,
                command_deadband_pct=1.0,
                time_fn=lambda: now[0],
            )

            self.assertTrue(dispatcher.send_wheel_speeds(80.0, -80.0))
            self.assertTrue(dispatcher.send_wheel_speeds(50.2, -49.8))
            now[0] += 0.2
            self.assertTrue(dispatcher.send_wheel_speeds(25.0, 20.0))
            dispatcher.close()

        self.assertEqual(controllers[0].commands, ["LR 50.0 -50.0", "LR 25.0 20.0"])
        self.assertTrue(controllers[0].closed)

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

    def test_near_zone_handoff_stops_tcp_tracking_then_turns_and_runs_tcp_move(self) -> None:
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
            events.append(("tcp_lr", left_pct, None))
            return True

        dispatcher.send_wheel_speeds = record_stop

        class FakeRobotController:
            def __init__(self, robot_ip: str, **_kwargs: object) -> None:
                self.robot_ip = robot_ip

            def turn(self, degrees: float, speed_percent: int) -> str:
                events.append(("turn", degrees, speed_percent))
                return "OK"

            def move(self, distance: float, speed_percent: int) -> str:
                events.append(("move", distance, speed_percent))
                return "OK"

        with patch("tools.topdown_object_detector.RobotController", FakeRobotController):
            owns_control = app.update_pickup_state(drive_runtime, now_s=1.0)
            app.runtime.robot_pose = RobotPose(
                x_cm=6.0,
                y_cm=0.0,
                heading_rad=0.0,
                tube_x_cm=0.0,
                tube_y_cm=0.0,
            )
            owns_control_after_vision = app.update_pickup_state(drive_runtime, now_s=1.1)
            app.runtime.near_zone_alignment_settle_started_s = (
                time.perf_counter() - app.config.drive.visual_servo_settle_time_s - 0.01
            )
            owns_control_after_settle = app.update_pickup_state(drive_runtime, now_s=1.2)

        self.assertTrue(owns_control)
        self.assertTrue(owns_control_after_vision)
        self.assertTrue(owns_control_after_settle)
        self.assertEqual(dispatcher.commands[-1][:2], (0.0, 0.0))
        self.assertEqual(events[0], ("tcp_lr", 0.0, None))
        self.assertEqual(events[1], ("turn", 10.0, int(round(app.config.drive.near_zone_turn_speed_pct))))
        self.assertIn(("move", 4.0, int(round(app.config.drive.near_zone_move_speed_pct))), events)
        self.assertEqual(app.runtime.balls_collected, 1)
        self.assertEqual(app.runtime.pickup_state, PickupExecutionState.PICKUP_ASSIST)

    def test_pickup_target_body_pose_uses_line_of_sight_not_stale_robot_heading(self) -> None:
        app = TopdownDetectorApp()
        app.runtime.robot_pose = RobotPose(
            x_cm=0.0,
            y_cm=3.0,
            heading_rad=math.radians(90.0),
            tube_x_cm=0.0,
            tube_y_cm=3.0,
        )
        app.runtime.latest_smoothed_balls = [
            SmoothedBallCoordinate(1, "white", (0, 0), (0, 0), 4, 12.0, 3.0),
        ]
        params = {"tube_forward_cm": 4.0, "tube_right_cm": 0.0}

        target_x_cm, target_y_cm = app._pickup_target_body_pose(
            HybridPose(8.0, 3.0, 0.0),
            params=params,
        )

        self.assertAlmostEqual(target_x_cm, 8.0)
        self.assertAlmostEqual(target_y_cm, 3.0)

    def test_near_zone_visual_servo_keeps_turning_until_noise_floor(self) -> None:
        app = TopdownDetectorApp()
        dispatcher = FakeDispatcher()
        drive_runtime = DriveRuntime(enabled=True, dispatcher=dispatcher)
        app.runtime.initial_total_balls = 1
        app.runtime.robot_pose = RobotPose(6.0, 0.0, 0.0, 0.0, 0.0)
        app.runtime.route_plan = RoutePlan(
            points=[HybridPose(0.0, 0.0, 0.0), HybridPose(10.0, 0.0, 0.0)],
            active_target=None,
            pickup_poses=[HybridPose(10.0, 0.0, 0.0)],
        )
        events = []

        class FakeRobotController:
            def __init__(self, robot_ip: str, **_kwargs: object) -> None:
                self.robot_ip = robot_ip

            def turn(self, degrees: float, speed_percent: int) -> str:
                events.append(("turn", degrees, speed_percent))
                return "OK"

            def move(self, distance: float, speed_percent: int) -> str:
                events.append(("move", distance, speed_percent))
                return "OK"

        with patch("tools.topdown_object_detector.RobotController", FakeRobotController):
            app.update_pickup_state(drive_runtime, now_s=1.0)
            app.runtime.robot_pose = RobotPose(6.0, 2.0, 0.0, 0.0, 0.0)
            app.update_pickup_state(drive_runtime, now_s=1.1)
            app.runtime.robot_pose = RobotPose(6.0, 0.3, 0.0, 0.0, 0.0)
            app.update_pickup_state(drive_runtime, now_s=1.2)
            app.runtime.robot_pose = RobotPose(6.0, 0.1, 0.0, 0.0, 0.0)
            app.update_pickup_state(drive_runtime, now_s=1.3)
            app.runtime.near_zone_alignment_settle_started_s = (
                time.perf_counter() - app.config.drive.visual_servo_settle_time_s - 0.01
            )
            app.update_pickup_state(drive_runtime, now_s=1.4)

        turn_events = [event for event in events if event[0] == "turn"]
        self.assertGreaterEqual(len(turn_events), 2)
        self.assertGreater(abs(turn_events[0][1]), abs(turn_events[-1][1]))
        self.assertGreater(turn_events[0][2], turn_events[-1][2])
        self.assertIn(("move", 4.0, int(round(app.config.drive.near_zone_move_speed_pct))), events)
        self.assertEqual(app.runtime.pickup_state, PickupExecutionState.PICKUP_ASSIST)

    def test_visual_servo_recorrects_when_settling_drifts_outside_noise_floor(self) -> None:
        app = TopdownDetectorApp()
        dispatcher = FakeDispatcher()
        drive_runtime = DriveRuntime(enabled=True, dispatcher=dispatcher)
        app.runtime.initial_total_balls = 1
        app.runtime.robot_pose = RobotPose(6.0, 0.0, 0.0, 0.0, 0.0)
        app.runtime.route_plan = RoutePlan(
            points=[HybridPose(0.0, 0.0, 0.0), HybridPose(10.0, 0.0, 0.0)],
            active_target=None,
            pickup_poses=[HybridPose(10.0, 0.0, 0.0)],
        )
        events = []

        class FakeRobotController:
            def __init__(self, robot_ip: str, **_kwargs: object) -> None:
                self.robot_ip = robot_ip

            def turn(self, degrees: float, speed_percent: int) -> str:
                events.append(("turn", degrees, speed_percent))
                return "OK"

            def move(self, distance: float, speed_percent: int) -> str:
                events.append(("move", distance, speed_percent))
                return "OK"

        with patch("tools.topdown_object_detector.RobotController", FakeRobotController):
            app.update_pickup_state(drive_runtime, now_s=1.0)
            app.runtime.robot_pose = RobotPose(6.0, 0.1, 0.0, 0.0, 0.0)
            app.update_pickup_state(drive_runtime, now_s=1.1)
            app.runtime.near_zone_alignment_settle_started_s = (
                time.perf_counter() - app.config.drive.visual_servo_settle_time_s - 0.01
            )
            app.runtime.robot_pose = RobotPose(6.0, 2.0, 0.0, 0.0, 0.0)
            app.update_pickup_state(drive_runtime, now_s=1.2)

        self.assertIn(("turn", 8.0, int(round(app.config.drive.near_zone_turn_speed_pct))), events)
        self.assertNotIn(("move", 4.0, int(round(app.config.drive.near_zone_move_speed_pct))), events)
        self.assertEqual(app.runtime.pickup_state, PickupExecutionState.NAVIGATION)

    def test_visual_servo_converges_when_alignment_stops_improving(self) -> None:
        app = TopdownDetectorApp()

        for _index in range(app.config.drive.visual_servo_stall_frames + 1):
            aligned, _turn_deg, _speed_pct, reason = app._near_zone_visual_servo_turn(4.0, 2.0)

        self.assertTrue(aligned)
        self.assertEqual(reason, "no further improvement")

    def test_autonomous_collection_runs_pickup_assist_not_full_unload(self) -> None:
        app = TopdownDetectorApp()
        dispatcher = FakeDispatcher()
        drive_runtime = DriveRuntime(enabled=True, dispatcher=dispatcher)
        app.runtime.pickup_state = PickupExecutionState.PICKUP_ASSIST
        events = []

        class FakeRobotController:
            def __init__(self, robot_ip: str, **_kwargs: object) -> None:
                self.robot_ip = robot_ip

            def pickup_assist(self) -> str:
                events.append("pickup_assist")
                return "OK"

            def unload_full_cycle(self) -> str:
                events.append("unload_full_cycle")
                return "OK"

            def turn(self, _degrees: float, _speed: int = 100) -> str:
                events.append("turn")
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
            def __init__(self, robot_ip: str, **_kwargs: object) -> None:
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

    def test_pickup_replan_state_keeps_cached_route_when_remaining_balls_match_plan(self) -> None:
        app = TopdownDetectorApp()
        drive_runtime = DriveRuntime(enabled=True, dispatcher=FakeDispatcher())
        app.runtime.initial_total_balls = 2
        app.runtime.balls_collected = 1
        app.runtime.pickup_state = PickupExecutionState.REPLAN
        app.runtime.robot_pose = RobotPose(10.0, 0.0, 0.0, 10.0, 0.0)
        app.runtime.route_plan = RoutePlan(
            points=[
                HybridPose(0.0, 0.0, 0.0),
                HybridPose(10.0, 0.0, 0.0),
                HybridPose(20.0, 0.0, 0.0),
            ],
            active_target=SimpleNamespace(track_id=1),
            pickup_poses=[HybridPose(10.0, 0.0, 0.0), HybridPose(20.0, 0.0, 0.0)],
        )
        app.runtime.route_cache_ball_signature = (
            (1, "white", 2, 0),
            (2, "white", 4, 0),
        )
        app.runtime.route_cache_unload_extension_cm = 15.0
        app.runtime.route_cache_target_id = 1
        app.runtime.latest_smoothed_balls = [
            SmoothedBallCoordinate(2, "white", (0, 0), (0, 0), 4, 20.0, 0.0),
        ]

        owns_control = app.update_pickup_state(drive_runtime, now_s=1.0)

        self.assertTrue(owns_control)
        self.assertTrue(app.runtime.route_plan.points)
        self.assertEqual(app.runtime.route_plan.pickup_poses, [HybridPose(20.0, 0.0, 0.0)])
        self.assertIsNone(app.runtime.route_plan.active_target)
        self.assertEqual(app.runtime.route_cache_target_id, -1)
        self.assertEqual(app.runtime.pickup_state, PickupExecutionState.NAVIGATION)
        self.assertAlmostEqual(app._pickup_distance_to_goal_cm(), 10.0)
        self.assertEqual(drive_runtime.last_message, "pickup complete; continuing cached route")

    def test_unload_endpoint_runs_pipe_only_double_unload_once(self) -> None:
        app = TopdownDetectorApp()
        dispatcher = FakeDispatcher()
        drive_runtime = DriveRuntime(enabled=True, dispatcher=dispatcher)
        app.runtime.collector_state = CollectorPositionState.TRAVEL
        app.runtime.initial_total_balls = 1
        app.runtime.balls_collected = 1
        unload_x = app.config.robot.tuned_footprint_rear_from_origin_cm + app.config.robot.tuned_unload_extension_cm
        unload_y = app.config.field.height_cm * 0.5
        app.runtime.robot_pose = RobotPose(unload_x, unload_y, 0.0, 0.0, 0.0)
        app.runtime.route_plan = RoutePlan(
            points=[HybridPose(unload_x + 10.0, unload_y, 0.0), HybridPose(unload_x, unload_y, 0.0)],
            active_target=None,
            pickup_poses=[],
            unload_pose=HybridPose(unload_x, unload_y, 0.0),
            unload_goal_cm=(0.0, unload_y),
        )
        events = []

        class FakeRobotController:
            def __init__(self, robot_ip: str, **_kwargs: object) -> None:
                self.robot_ip = robot_ip

            def unload_full_cycle(self) -> str:
                events.append("unload_full_cycle")
                return "OK"

            def turn(self, _degrees: float, _speed: int = 100) -> str:
                events.append("turn")
                return "OK"

            def pipe_down(self, units: float, speed: int) -> str:
                events.append(f"pipe_down {units} {speed}")
                return "OK"

            def pipe_up(self, units: float, speed: int) -> str:
                events.append(f"pipe_up {units} {speed}")
                return "OK"

            def pipe_stop(self) -> str:
                events.append("pipe_stop")
                return "OK"

            def move(self, _distance: float, _speed: int) -> str:
                events.append("move")
                return "unexpected"

            def back(self, _distance: float, _speed: int) -> str:
                events.append("back")
                return "unexpected"

        with patch("tools.topdown_object_detector.RobotController", FakeRobotController):
            owns_control = app.update_unload_state(drive_runtime, now_s=1.0)
            self.assertIsNone(app.unload_thread)
            self.assertEqual(app.runtime.unload_state, UnloadExecutionState.PRE_UNLOAD_PIVOT)
            owns_after_pivot = app.update_unload_state(drive_runtime, now_s=1.05)
            self.assertIsNone(app.unload_thread)
            app.runtime.unload_alignment_settle_started_s = (
                time.perf_counter() - app.config.drive.visual_servo_settle_time_s - 0.01
            )
            owns_after_alignment = app.update_unload_state(drive_runtime, now_s=1.1)
            assert app.unload_thread is not None
            app.unload_thread.join(timeout=1.0)
            owns_after_complete = app.update_unload_state(drive_runtime, now_s=1.2)

        self.assertTrue(owns_control)
        self.assertTrue(owns_after_pivot)
        self.assertTrue(owns_after_alignment)
        self.assertTrue(owns_after_complete)
        self.assertEqual(
            events,
            ["unload_full_cycle", "pipe_down 2.0 35", "pipe_up 2.0 35", "unload_full_cycle", "pipe_stop"],
        )
        self.assertNotIn("move", events)
        self.assertNotIn("back", events)
        self.assertEqual(app.runtime.unload_state, UnloadExecutionState.COMPLETE)
        self.assertEqual(dispatcher.commands[0][:2], (0.0, 0.0))

        with patch("tools.topdown_object_detector.RobotController", FakeRobotController):
            app.update_unload_state(drive_runtime, now_s=2.0)

        self.assertEqual(
            events,
            ["unload_full_cycle", "pipe_down 2.0 35", "pipe_up 2.0 35", "unload_full_cycle", "pipe_stop"],
        )

    def test_unload_visual_servo_recorrects_instead_of_dropping_after_settle_drift(self) -> None:
        app = TopdownDetectorApp()
        dispatcher = FakeDispatcher()
        drive_runtime = DriveRuntime(enabled=True, dispatcher=dispatcher)
        app.runtime.initial_total_balls = 1
        app.runtime.balls_collected = 1
        unload_x = app.config.robot.tuned_footprint_rear_from_origin_cm + app.config.robot.tuned_unload_extension_cm
        unload_y = app.config.field.height_cm * 0.5
        app.runtime.robot_pose = RobotPose(unload_x, unload_y, 0.0, 0.0, 0.0)
        app.runtime.route_plan = RoutePlan(
            points=[HybridPose(unload_x + 10.0, unload_y, 0.0), HybridPose(unload_x, unload_y, 0.0)],
            active_target=None,
            pickup_poses=[],
            unload_pose=HybridPose(unload_x, unload_y, 0.0),
            unload_goal_cm=(0.0, unload_y),
        )
        events = []

        class FakeRobotController:
            def __init__(self, robot_ip: str, **_kwargs: object) -> None:
                self.robot_ip = robot_ip

            def back(self, distance: float, speed_percent: int) -> str:
                events.append(("back", distance, speed_percent))
                return "OK"

            def move(self, distance: float, speed_percent: int) -> str:
                events.append(("move", distance, speed_percent))
                return "OK"

            def turn(self, degrees: float, speed_percent: int) -> str:
                events.append(("turn", degrees, speed_percent))
                return "OK"

            def unload_full_cycle(self) -> str:
                events.append(("unload_full_cycle",))
                return "unexpected"

        with patch("tools.topdown_object_detector.RobotController", FakeRobotController):
            self.assertTrue(app.update_unload_state(drive_runtime, now_s=1.0))
            self.assertEqual(app.runtime.unload_state, UnloadExecutionState.PRE_UNLOAD_PIVOT)
            self.assertTrue(app.update_unload_state(drive_runtime, now_s=1.05))
            app.runtime.unload_alignment_settle_started_s = (
                time.perf_counter() - app.config.drive.visual_servo_settle_time_s - 0.01
            )
            app.runtime.robot_pose = RobotPose(unload_x + 5.0, unload_y, 0.0, 0.0, 0.0)
            self.assertTrue(app.update_unload_state(drive_runtime, now_s=1.1))

        self.assertIsNone(app.unload_thread)
        self.assertEqual(app.runtime.unload_state, UnloadExecutionState.ALIGNING)
        self.assertIn(("back", 3.0, int(round(app.config.drive.near_zone_move_speed_pct))), events)
        self.assertNotIn(("unload_full_cycle",), events)

    def test_unload_staging_pivot_suspends_xte_and_tank_turns(self) -> None:
        app = TopdownDetectorApp()
        dispatcher = FakeDispatcher()
        drive_runtime = DriveRuntime(enabled=True, dispatcher=dispatcher)
        app.runtime.initial_total_balls = 1
        app.runtime.balls_collected = 1
        unload_y = app.config.field.height_cm * 0.5
        app.runtime.robot_pose = RobotPose(70.0, unload_y, math.radians(90.0), 0.0, 0.0)
        app.runtime.route_plan = RoutePlan(
            points=[HybridPose(90.0, unload_y, 0.0), HybridPose(50.0, unload_y, 0.0)],
            active_target=None,
            pickup_poses=[],
            unload_pose=HybridPose(50.0, unload_y, 0.0),
            unload_goal_cm=(0.0, unload_y),
        )

        owns_control = app.update_unload_state(drive_runtime, now_s=1.0)
        pivot_owns_control = app.update_unload_state(drive_runtime, now_s=1.1)

        self.assertTrue(owns_control)
        self.assertTrue(pivot_owns_control)
        self.assertEqual(app.runtime.unload_state, UnloadExecutionState.PRE_UNLOAD_PIVOT)
        self.assertEqual(drive_runtime.state, DriveControlState.PRE_UNLOAD_PIVOT)
        self.assertEqual(dispatcher.commands[0][:2], (0.0, 0.0))
        self.assertEqual(
            dispatcher.commands[-1],
            (
                app.config.drive.unload_pivot_speed_pct,
                -app.config.drive.unload_pivot_speed_pct,
                True,
            ),
        )

    def test_unload_endpoint_waits_for_optimistic_collection_complete(self) -> None:
        app = TopdownDetectorApp()
        dispatcher = FakeDispatcher()
        drive_runtime = DriveRuntime(enabled=True, dispatcher=dispatcher)
        app.runtime.initial_total_balls = 2
        app.runtime.balls_collected = 1
        app.runtime.robot_pose = RobotPose(25.0, 60.0, 0.0, 42.1, 60.0)
        app.runtime.route_plan = RoutePlan(
            points=[HybridPose(40.0, 60.0, 0.0), HybridPose(25.0, 60.0, 0.0)],
            active_target=None,
            pickup_poses=[],
            unload_pose=HybridPose(25.0, 60.0, 0.0),
            unload_goal_cm=(0.0, 60.0),
        )

        owns_control = app.update_unload_state(drive_runtime, now_s=1.0)

        self.assertFalse(owns_control)
        self.assertFalse(app.runtime.unload_command_fired)
        self.assertIsNone(app.unload_thread)
        self.assertEqual(dispatcher.commands, [])

    def test_manual_wheel_keys_are_blocked_during_unload_but_stop_still_works(self) -> None:
        app = TopdownDetectorApp()
        dispatcher = FakeDispatcher()
        drive_runtime = DriveRuntime(enabled=True, dispatcher=dispatcher)
        app.runtime.unload_state = UnloadExecutionState.PIPE_SHAKE

        app.handle_manual_robot_key(next(iter(app.config.drive.key_up_arrow)), drive_runtime)
        app.handle_manual_robot_key(ord(" "), drive_runtime)

        self.assertEqual(len(dispatcher.commands), 1)
        self.assertEqual(dispatcher.commands[0][:2], (0.0, 0.0))
        self.assertEqual(drive_runtime.last_message, "manual stop")

    def test_stale_async_route_result_is_rejected_when_start_pose_moves(self) -> None:
        app = TopdownDetectorApp()
        ball_signature = ((1, "white", 100, 200),)
        old_start = HybridPose(10.0, 10.0, 0.0)
        new_start = HybridPose(40.0, 10.0, 0.0)
        route_plan = RoutePlan(
            points=[old_start, HybridPose(20.0, 10.0, 0.0)],
            active_target=None,
            pickup_poses=[],
        )
        request = RoutePlanningRequest(
            request_id=1,
            grid=SimpleNamespace(copy=lambda: None),
            ball_targets=[],
            start_pose=old_start,
            geometry=RobotGeometry(20.0, 8.0, 10.0, 17.0, 0.0),
            ball_signature=ball_signature,
            start_signature=app.route_start_signature(old_start),
            unload_extension_cm=15.0,
        )
        completed = RoutePlanningResult(request=request, route_plan=route_plan, duration_ms=1.0)
        app.async_route_planner = SimpleNamespace(poll_completed=lambda: completed)

        app.accept_completed_route_if_current(
            ball_signature,
            15.0,
            app.route_start_signature(new_start),
        )

        self.assertEqual(app.runtime.route_plan.points, [])

    def test_final_pickup_preserves_existing_unload_route_when_collection_complete(self) -> None:
        app = TopdownDetectorApp()
        drive_runtime = DriveRuntime(enabled=True, dispatcher=FakeDispatcher())
        app.runtime.initial_total_balls = 1
        app.runtime.balls_collected = 1
        app.runtime.pickup_state = PickupExecutionState.REPLAN
        app.runtime.robot_pose = RobotPose(10.0, 0.0, 0.0, 10.0, 0.0)
        app.runtime.route_plan = RoutePlan(
            points=[HybridPose(10.0, 0.0, 0.0), HybridPose(25.0, 60.0, 0.0)],
            active_target=None,
            pickup_poses=[HybridPose(10.0, 0.0, 0.0)],
            unload_pose=HybridPose(25.0, 60.0, 0.0),
            unload_goal_cm=(0.0, 60.0),
        )

        owns_control = app.update_pickup_state(drive_runtime, now_s=1.0)

        self.assertTrue(owns_control)
        self.assertTrue(app.runtime.route_plan.points)
        self.assertIsNotNone(app.runtime.route_plan.unload_pose)
        self.assertEqual(app.runtime.route_plan.pickup_poses, [])
        self.assertEqual(app.runtime.pickup_state, PickupExecutionState.NAVIGATION)
        self.assertEqual(drive_runtime.last_message, "pickup complete; routing to unload")


if __name__ == "__main__":
    unittest.main()
