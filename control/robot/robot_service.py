"""Process boundary for robot control.

The robot service owns the real RobotCommander and its EV3 TCP connection.
GUI and planning code talk to it through a lightweight proxy so rendering and
vision work cannot block the motor command path.
"""

from __future__ import annotations

from dataclasses import dataclass
from queue import Empty
import multiprocessing as mp
from typing import Any
import uuid

from control.commander import RobotCommander
from config import ConnectionConfig, DriveConfig


@dataclass(frozen=True)
class RobotCommand:
    """Single command crossing the GUI/service boundary."""

    command_id: str
    method: str
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] | None = None
    wait_for_result: bool = False


@dataclass(frozen=True)
class RobotCommandResult:
    """Reply from the robot service for a command that requested one."""

    command_id: str
    ok: bool
    result: Any = None
    error: str | None = None


class RemoteRobotCommander:
    """Commander-compatible proxy that forwards requests to a robot service."""

    def __init__(
        self,
        request_queue: mp.Queue[RobotCommand | None],
        response_queue: mp.Queue[RobotCommandResult] | None = None,
        *_,
        **__,
    ) -> None:
        self._request_queue = request_queue
        self._response_queue = response_queue
        self.last_error: str = ""
        self.sock: object | None = object()
        self._closed = False

    def _submit(self, method: str, *args: Any, wait_for_result: bool = False, **kwargs: Any) -> Any:
        if self._closed:
            self.last_error = "remote commander is closed"
            return False if not wait_for_result else None

        command_id = uuid.uuid4().hex
        payload = RobotCommand(
            command_id=command_id,
            method=method,
            args=tuple(args),
            kwargs=dict(kwargs) if kwargs else None,
            wait_for_result=wait_for_result,
        )
        self._request_queue.put(payload)

        if not wait_for_result:
            return True

        if self._response_queue is None:
            self.last_error = "remote commander has no response queue"
            return None

        while True:
            try:
                reply = self._response_queue.get(timeout=5.0)
            except Empty:
                self.last_error = f"timed out waiting for {method}"
                return None
            if reply.command_id != command_id:
                continue
            if reply.ok:
                self.last_error = ""
                return reply.result
            self.last_error = reply.error or f"{method} failed"
            return None

    def turn(self, degrees: float) -> bool:
        return bool(self._submit("turn", degrees))

    def drive(self, cm: float, dt_s: float | None = None) -> bool:
        return bool(self._submit("drive", cm, dt_s))

    def drive_adjusted(self, cm: float, dt_s: float, heading_deg: float) -> bool:
        return bool(self._submit("drive_adjusted", cm, dt_s, heading_deg))

    def stop(self, force: bool = False) -> bool:
        return bool(self._submit("stop", force=force))

    def pickup(self) -> bool:
        return bool(self._submit("pickup", wait_for_result=True))

    def dropoff(self) -> bool:
        return bool(self._submit("dropoff", wait_for_result=True))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.sock = None


class RobotControlService:
    """Own the real RobotCommander in a dedicated process."""

    def __init__(self, connection_config: ConnectionConfig, drive_config: DriveConfig) -> None:
        self._context = mp.get_context("spawn")
        self._request_queue: mp.Queue[RobotCommand | None] = self._context.Queue()
        self._response_queue: mp.Queue[RobotCommandResult] = self._context.Queue()
        self._status_queue: mp.Queue[tuple[str, str | None]] = self._context.Queue()
        self._process: mp.Process | None = None
        self._connection_config = connection_config
        self._drive_config = drive_config

    @property
    def request_queue(self) -> mp.Queue[RobotCommand | None]:
        return self._request_queue

    @property
    def response_queue(self) -> mp.Queue[RobotCommandResult]:
        return self._response_queue

    @property
    def status_queue(self) -> mp.Queue[tuple[str, str | None]]:
        return self._status_queue

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.is_alive()

    def start(self) -> None:
        if self.running:
            return
        self._process = self._context.Process(
            target=_robot_service_main,
            args=(self._request_queue, self._response_queue, self._status_queue, self._connection_config, self._drive_config),
            daemon=True,
        )
        self._process.start()

    def close(self) -> None:
        try:
            self._request_queue.put_nowait(None)
        except Exception:
            pass
        if self._process is not None:
            self._process.join(timeout=2.0)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=1.0)
            self._process = None


def _robot_service_main(
    request_queue: mp.Queue[RobotCommand | None],
    response_queue: mp.Queue[RobotCommandResult],
    status_queue: mp.Queue[tuple[str, str | None]],
    connection_config: ConnectionConfig,
    drive_config: DriveConfig,
) -> None:
    commander: RobotCommander | None = None
    try:
        commander = RobotCommander(
            connection_config=connection_config,
            drive_config=drive_config,
            auto_connect=True,
        )
        status_queue.put(("ready", None))
    except Exception as exc:  # pragma: no cover - defensive startup guard
        status_queue.put(("error", str(exc)))

    while True:
        request = request_queue.get()
        if request is None:
            break
        if request.method == "close":
            break

        if commander is None:
            response_queue.put(
                RobotCommandResult(
                    command_id=request.command_id,
                    ok=False,
                    error="robot commander unavailable",
                )
            )
            continue

        try:
            method = getattr(commander, request.method)
            kwargs = request.kwargs or {}
            result = method(*request.args, **kwargs)
            if request.wait_for_result:
                response_queue.put(RobotCommandResult(command_id=request.command_id, ok=True, result=result))
        except Exception as exc:  # pragma: no cover - defensive service guard
            if request.wait_for_result:
                response_queue.put(RobotCommandResult(command_id=request.command_id, ok=False, error=str(exc)))
            else:
                response_queue.put(RobotCommandResult(command_id=request.command_id, ok=False, error=str(exc)))

    if commander is not None:
        try:
            commander.stop(force=True)
        except Exception:
            pass
        try:
            commander.close()
        except Exception:
            pass
    status_queue.put(("closed", None))
