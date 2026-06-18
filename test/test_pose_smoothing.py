"""Unit tests for EMA pose smoothing in RobotPoseEstimator.

Covers the noise-rejection helper directly (deterministic, no camera/ArUco):
- linear EMA on position
- wraparound-safe cos/sin EMA on heading
- first-sample seeding (no startup lag)
- reset-on-dropout snapping to a fresh measurement
- enable flag passthrough
"""

from __future__ import annotations

import math

import pytest

from config import PoseSmoothingConfig
from localization.localization import RobotPoseEstimator, normalize_angle


def _estimator(**kwargs) -> RobotPoseEstimator:
    return RobotPoseEstimator(smoothing_config=PoseSmoothingConfig(**kwargs))


class TestSmoothPose:
    def test_first_sample_is_passed_through(self):
        est = _estimator(position_alpha=0.3, heading_alpha=0.3)
        x, y, h = est._smooth_pose(10.0, 20.0, 0.5)
        assert (x, y, h) == pytest.approx((10.0, 20.0, 0.5))

    def test_position_converges_toward_steady_value(self):
        est = _estimator(position_alpha=0.3, heading_alpha=0.3)
        est._smooth_pose(0.0, 0.0, 0.0)  # seed
        last_x = 0.0
        for _ in range(100):
            last_x, _y, _h = est._smooth_pose(100.0, 0.0, 0.0)
        assert last_x == pytest.approx(100.0, abs=1e-3)

    def test_single_step_blend_matches_formula(self):
        est = _estimator(position_alpha=0.25, heading_alpha=0.25)
        est._smooth_pose(0.0, 0.0, 0.0)  # seed at origin
        x, y, _h = est._smooth_pose(40.0, 80.0, 0.0)
        # smoothed = prev + alpha*(raw - prev)
        assert x == pytest.approx(0.25 * 40.0)
        assert y == pytest.approx(0.25 * 80.0)

    def test_stationary_noisy_input_is_dampened(self):
        """Jitter around a fixed pose should produce a far steadier output."""
        est = _estimator(position_alpha=0.3, heading_alpha=0.3)
        noise = [+1.0, -1.0, +1.0, -1.0, +1.0, -1.0, +1.0, -1.0]
        est._smooth_pose(50.0, 50.0, 0.0)  # seed at the true pose
        outputs = [est._smooth_pose(50.0 + n, 50.0, 0.0)[0] for n in noise]
        # Raw swing is 2.0 cm peak-to-peak; smoothed swing must be much smaller.
        assert max(outputs) - min(outputs) < 1.0

    def test_heading_wraps_across_pi_seam(self):
        """Averaging 179 deg and -179 deg must land near +/-180, not near 0."""
        est = _estimator(position_alpha=0.5, heading_alpha=0.5)
        est._smooth_pose(0.0, 0.0, math.radians(179.0))  # seed
        _x, _y, h = est._smooth_pose(0.0, 0.0, math.radians(-179.0))
        assert abs(abs(h) - math.pi) < math.radians(2.0)

    def test_heading_converges_toward_steady_value(self):
        est = _estimator(position_alpha=0.3, heading_alpha=0.3)
        target = math.radians(120.0)
        est._smooth_pose(0.0, 0.0, 0.0)  # seed
        h = 0.0
        for _ in range(100):
            _x, _y, h = est._smooth_pose(0.0, 0.0, target)
        assert h == pytest.approx(target, abs=1e-3)

    def test_reset_makes_next_sample_seed_again(self):
        est = _estimator(position_alpha=0.3, heading_alpha=0.3)
        est._smooth_pose(0.0, 0.0, 0.0)
        est._smooth_pose(100.0, 0.0, 0.0)  # filter now mid-converge
        est._reset_smoothing()
        # After a dropout the filter must snap to the fresh measurement.
        x, y, h = est._smooth_pose(500.0, 250.0, 1.0)
        assert (x, y, h) == pytest.approx((500.0, 250.0, 1.0))

    def test_disabled_is_identity(self):
        est = _estimator(enabled=False, position_alpha=0.3, heading_alpha=0.3)
        est._smooth_pose(0.0, 0.0, 0.0)
        x, y, h = est._smooth_pose(123.0, 456.0, 0.7)
        assert (x, y, h) == pytest.approx((123.0, 456.0, 0.7))

    def test_output_heading_is_normalized(self):
        est = _estimator(position_alpha=0.5, heading_alpha=0.5)
        est._smooth_pose(0.0, 0.0, math.radians(170.0))
        _x, _y, h = est._smooth_pose(0.0, 0.0, math.radians(-170.0))
        assert h == pytest.approx(normalize_angle(h))
        assert -math.pi <= h < math.pi
