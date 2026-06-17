"""Unit tests for the spin-calibration backend.

Covers:
- control.spin_calibration.SpinController termination (full sweep + time cap)
- localization single-marker tolerance in compute_spin_centers
- robot_calibration.json geometry round-trip and merge-on-save
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from control.spin_calibration import SpinController, SpinStatus
from localization.localization import RobotCalibrationCollector
from localization.models import RobotCalibrationRuntime, RobotGeometry, RobotMarkerObservation
from perception.vision.models import ParallaxConfig
from config import RobotCalibrationConfig, RobotGeometryConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _StubCommander:
    """Records steer/stop calls so spin behaviour can be asserted."""

    def __init__(self) -> None:
        self.steer_calls: list[tuple[float, float, bool]] = []
        self.stop_calls = 0

    def steer(self, base_speed_pct: float, turn_speed_pct: float, force: bool = False) -> bool:
        self.steer_calls.append((base_speed_pct, turn_speed_pct, force))
        return True

    def stop(self, force: bool = True) -> bool:
        self.stop_calls += 1
        return True


def _fake_clock(start: float = 0.0, step: float = 0.0):
    t = [start]

    def now() -> float:
        value = t[0]
        t[0] += step
        return value

    return now, t


def _observation(marker_id: int, ground_center: tuple[float, float], yaw_rad: float) -> RobotMarkerObservation:
    center = np.array(ground_center, dtype=np.float32)
    corners = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    return RobotMarkerObservation(
        marker_id=marker_id,
        center=center,
        ground_center=center,
        corners=corners,
        ground_corners=corners,
        yaw_rad=yaw_rad,
    )


def _ring_points(center: tuple[float, float], radius: float, count: int) -> list[tuple[float, float]]:
    cx, cy = center
    return [
        (cx + radius * math.cos(2 * math.pi * i / count), cy + radius * math.sin(2 * math.pi * i / count))
        for i in range(count)
    ]


# ---------------------------------------------------------------------------
# SpinController
# ---------------------------------------------------------------------------

class TestSpinController:
    def test_completes_after_target_sweep(self):
        commander = _StubCommander()
        spin = SpinController(commander, target_sweep_deg=360.0, max_spin_s=100.0)
        spin.start()

        # Feed monotonically increasing yaw in small steps for both markers.
        status = SpinStatus.SPINNING
        yaw = 0.0
        for _ in range(400):
            yaw += math.radians(5.0)
            wrapped = (yaw + math.pi) % (2 * math.pi) - math.pi
            status = spin.tick({4: wrapped, 5: wrapped})
            if status is SpinStatus.COMPLETE:
                break

        assert status is SpinStatus.COMPLETE
        assert spin.swept_deg >= 360.0
        assert commander.stop_calls >= 1
        assert not spin.active

    def test_keeps_spinning_before_target(self):
        commander = _StubCommander()
        spin = SpinController(commander, target_sweep_deg=360.0, max_spin_s=100.0)
        spin.start()
        status = spin.tick({4: 0.0})
        status = spin.tick({4: math.radians(30.0)})
        assert status is SpinStatus.SPINNING
        assert spin.active
        # Continued to command an in-place rotation (turn speed > 0, base = 0).
        assert commander.steer_calls[-1][0] == 0.0
        assert commander.steer_calls[-1][1] > 0.0

    def test_time_cap_stops_when_no_markers(self):
        now, _t = _fake_clock(start=0.0, step=20.0)  # each call advances 20s
        commander = _StubCommander()
        spin = SpinController(commander, max_spin_s=15.0, time_fn=now)
        spin.start()  # consumes one clock read (t=0)
        # Never see a marker; the time cap must end the spin.
        status = spin.tick({})
        assert status is SpinStatus.TIMED_OUT
        assert commander.stop_calls >= 1
        assert not spin.active

    def test_marker_dropout_does_not_inject_jump(self):
        commander = _StubCommander()
        spin = SpinController(commander, target_sweep_deg=360.0, max_spin_s=100.0)
        spin.start()
        spin.tick({4: 0.0, 5: 0.0})
        spin.tick({4: math.radians(10.0)})          # marker 5 disappears
        spin.tick({4: math.radians(20.0), 5: math.radians(20.0)})  # 5 returns far away
        # Only consecutive-tick deltas count; the re-seed of 5 adds nothing.
        assert spin.swept_deg == pytest.approx(20.0, abs=1e-3)


# ---------------------------------------------------------------------------
# compute_spin_centers single-marker tolerance
# ---------------------------------------------------------------------------

class TestComputeSpinCenters:
    def _collector(self) -> RobotCalibrationCollector:
        return RobotCalibrationCollector(
            RobotGeometryConfig(marker_ids=(4, 5)),
            RobotCalibrationConfig(min_robot_spin_points=20),
        )

    def test_both_markers_fitted(self):
        collector = self._collector()
        runtime = RobotCalibrationRuntime(
            collected_points={
                4: _ring_points((100.0, 100.0), 30.0, 24),
                5: _ring_points((100.0, 100.0), 50.0, 24),
            }
        )
        assert collector.compute_spin_centers(runtime) is True
        assert set(runtime.fitted_centers) == {4, 5}
        for cx, cy in runtime.fitted_centers.values():
            assert cx == pytest.approx(100.0, abs=2.0)
            assert cy == pytest.approx(100.0, abs=2.0)

    def test_single_marker_still_succeeds(self):
        collector = self._collector()
        runtime = RobotCalibrationRuntime(
            collected_points={
                4: _ring_points((80.0, 60.0), 40.0, 24),
                5: [(0.0, 0.0)] * 3,  # too few points
            }
        )
        assert collector.compute_spin_centers(runtime) is True
        assert set(runtime.fitted_centers) == {4}
        assert "Skipped" in runtime.warning

    def test_no_markers_fails(self):
        collector = self._collector()
        runtime = RobotCalibrationRuntime(
            collected_points={4: [(0.0, 0.0)] * 2, 5: [(0.0, 0.0)] * 2}
        )
        assert collector.compute_spin_centers(runtime) is False
        assert not runtime.fitted_centers
        assert "at least one marker" in runtime.warning


# ---------------------------------------------------------------------------
# Geometry persistence + merge-on-save
# ---------------------------------------------------------------------------

class TestGeometryPersistence:
    def _parallax(self) -> ParallaxConfig:
        return ParallaxConfig(
            marker_height_cm=9.0,
            camera_height_cm=170.0,
            calibration_plane_height_cm=9.0,
            camera_center=np.array([500.0, 400.0], dtype=np.float32),
        )

    def _geometry(self) -> RobotGeometry:
        return RobotGeometry(
            width_cm=20.0, front_cm=8.0, rear_cm=10.0,
            tube_forward_cm=17.0, tube_right_cm=0.0, unload_extension_cm=15.0,
        )

    def test_save_geometry_tuning_roundtrip(self, tmp_path: Path):
        collector = RobotCalibrationCollector(RobotGeometryConfig(marker_ids=(4, 5)))
        path = tmp_path / "robot_calibration.json"
        collector.save_geometry_tuning(path, self._geometry(), heading_tuning_rad=math.radians(3.0))

        data = json.loads(path.read_text())
        geom = data["geometry"]
        assert geom["width_cm"] == pytest.approx(20.0)
        assert geom["front_cm"] == pytest.approx(8.0)
        assert geom["heading_tuning_deg"] == pytest.approx(3.0, abs=1e-6)

        params: dict[str, object] = {}
        RobotCalibrationCollector.apply_geometry_to_params(params, geom)
        assert params["robot_front_cm"] == pytest.approx(8.0)
        assert params["heading_tuning_rad"] == pytest.approx(math.radians(3.0), abs=1e-9)

    def test_save_geometry_preserves_existing_markers(self, tmp_path: Path):
        collector = RobotCalibrationCollector(RobotGeometryConfig(marker_ids=(4, 5)))
        path = tmp_path / "robot_calibration.json"
        path.write_text(json.dumps({"markers": {"4": {"dx": 1.0, "dy": 2.0}}}))

        collector.save_geometry_tuning(path, self._geometry())
        data = json.loads(path.read_text())
        assert data["markers"]["4"]["dx"] == 1.0  # untouched
        assert "geometry" in data

    def test_save_calibration_merges_unfit_marker(self, tmp_path: Path):
        collector = RobotCalibrationCollector(RobotGeometryConfig(marker_ids=(4, 5)))
        path = tmp_path / "robot_calibration.json"
        # Pre-existing calibration for marker 5.
        path.write_text(json.dumps({"markers": {"5": {"dx": 9.0, "dy": 9.0, "alpha_rad": 0.0}}}))

        runtime = RobotCalibrationRuntime(fitted_centers={4: (100.0, 100.0)})
        observations = {4: _observation(4, (130.0, 100.0), yaw_rad=0.0)}

        result = collector.save_robot_calibration(
            path, runtime, observations, self._parallax(), (1000, 800),
            geometry=self._geometry(),
        )
        # Marker 4 freshly written, marker 5 preserved from disk.
        assert result["markers"]["4"]["dx"] == pytest.approx(30.0)
        assert result["markers"]["5"]["dx"] == 9.0
        assert "geometry" in result
