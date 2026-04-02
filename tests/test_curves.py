"""
Tests for gamma_curve and basal_curve generation utilities.

Verifies that:
- gamma_curve produces arrays whose sum equals total_amount
- basal_curve produces a trapezoidal curve whose sum equals total_amount
- Both curves have correct shapes and non-negative values
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulator import gamma_curve, basal_curve, DT_MINUTES


class TestGammaCurve:
    def test_sum_equals_total_amount(self):
        """Area (sum of per-step values) equals total_amount."""
        for total in [10.0, 40.0, 100.0]:
            curve = gamma_curve(total, k=2.0, theta=15.0, duration_minutes=120.0)
            assert abs(np.sum(curve) - total) < 1e-6, (
                f"Sum {np.sum(curve):.6f} != total_amount {total}")

    def test_all_non_negative(self):
        """Curve values are all non-negative."""
        curve = gamma_curve(40.0, k=2.0, theta=15.0, duration_minutes=120.0)
        assert np.all(curve >= 0), "Gamma curve contains negative values"

    def test_correct_length(self):
        """Curve length matches duration / DT_MINUTES."""
        duration = 120.0
        curve = gamma_curve(40.0, k=2.0, theta=15.0, duration_minutes=duration)
        expected_steps = int(duration / DT_MINUTES)
        assert len(curve) == expected_steps

    def test_fast_vs_slow_peak_timing(self):
        """Fast carb curve peaks earlier than slow carb curve."""
        fast = gamma_curve(40.0, k=2.0, theta=15.0, duration_minutes=200.0)
        slow = gamma_curve(40.0, k=4.0, theta=20.0, duration_minutes=400.0)
        assert np.argmax(fast) < np.argmax(slow), (
            "Fast curve should peak before slow curve")

    def test_zero_duration_returns_array(self):
        """Zero or negative duration returns a single-element array."""
        curve = gamma_curve(10.0, k=2.0, theta=15.0, duration_minutes=0.0)
        assert len(curve) >= 1

    def test_sum_invariant_to_shape_params(self):
        """Sum equals total_amount regardless of k and theta."""
        total = 55.0
        for k, theta in [(1.5, 10.0), (3.0, 25.0), (6.0, 30.0), (8.0, 50.0)]:
            duration = k * theta * 5
            curve = gamma_curve(total, k=k, theta=theta, duration_minutes=duration)
            assert abs(np.sum(curve) - total) < 1e-6, (
                f"k={k}, theta={theta}: sum {np.sum(curve):.6f} != {total}")


class TestBasalCurve:
    def test_sum_equals_total_amount(self):
        """Basal curve sum equals total_amount."""
        total = 20.0
        curve = basal_curve(total_amount=total, duration_minutes=1560.0)
        assert abs(np.sum(curve) - total) < 1e-6, (
            f"Sum {np.sum(curve):.6f} != total_amount {total}")

    def test_all_non_negative(self):
        """Basal curve values are all non-negative."""
        curve = basal_curve(total_amount=20.0, duration_minutes=1560.0)
        assert np.all(curve >= 0), "Basal curve contains negative values"

    def test_correct_length(self):
        """Curve length matches duration / DT_MINUTES."""
        duration = 1560.0
        curve = basal_curve(total_amount=20.0, duration_minutes=duration)
        assert len(curve) == int(duration / DT_MINUTES)

    def test_trapezoidal_shape(self):
        """Basal curve ramps up and ramps down (trapezoidal shape)."""
        curve = basal_curve(total_amount=20.0, duration_minutes=1560.0,
                            ramp_up_hours=3.0, ramp_down_hours=4.0)
        # First value should be less than the middle plateau
        mid = len(curve) // 2
        assert curve[0] < curve[mid], "Curve should ramp up from zero"
        # Last value should be less than middle plateau
        assert curve[-1] < curve[mid], "Curve should ramp down to zero"

    def test_units_compatible_with_gamma_curve(self):
        """Both curves produce values in amount-per-step units (not rate units)."""
        # gamma_curve: sum = total_amount
        # basal_curve: sum = total_amount
        # Both should be in grams-per-step so they can be directly added
        gamma = gamma_curve(40.0, k=2.0, theta=15.0, duration_minutes=120.0)
        basal = basal_curve(total_amount=20.0, duration_minutes=1560.0)
        # Verify magnitudes are comparable (both are grams per 5-min step)
        # gamma: 40g over 120min = 24 steps → ~1.67g/step average
        # basal: 20g over 1560min = 312 steps → ~0.064g/step average
        assert gamma.mean() < 10.0 and basal.mean() < 10.0, (
            "Curve values appear to be in wrong units (too large)")
