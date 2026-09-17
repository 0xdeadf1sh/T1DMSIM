"""Tests gamma_curve and basal_curve: sum equals total_amount (basal_curve
is a Bateman PK curve), correct shapes, non-negative values."""

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

    def test_bateman_shape(self):
        """Basal curve is a Bateman PK: a smooth rise from ~0 to a single broad
        interior peak near tmax (~5.3h), then a decline with the tail clipped
        back toward zero.
        """
        curve = basal_curve(total_amount=20.0, duration_minutes=1560.0)
        peak = int(np.argmax(curve))
        steps_per_hour = 60 // DT_MINUTES
        # Single broad interior peak near the analytic tmax, not a plateau or boundary spike.
        assert 4 * steps_per_hour < peak < 10 * steps_per_hour, (
            f"peak at step {peak} should sit near tmax≈5.3h")
        # Rises from ~0 and the smootherstep tail clips back toward 0.
        assert curve[0] < 0.2 * curve[peak], "curve should rise from ~0"
        assert curve[-1] < 0.2 * curve[peak], "tail-clip should taper toward 0"
        # Unimodal: monotone up to the peak, monotone down after it.
        assert np.all(np.diff(curve[:peak + 1]) >= -1e-9), "should rise to the peak"
        assert np.all(np.diff(curve[peak:]) <= 1e-9), "should fall after the peak"

    def test_amount_semantics_not_rate(self):
        """total_amount is an area, not a rate: sum is invariant to duration, so
        per-step mean scales down as duration grows. Catches a units/hour rate
        passed where an amount is expected (sum would then scale with duration)."""
        short = gamma_curve(40.0, k=2.0, theta=15.0, duration_minutes=120.0)
        long = gamma_curve(40.0, k=2.0, theta=15.0, duration_minutes=600.0)
        assert abs(short.sum() - 40.0) < 1e-6
        assert abs(long.sum() - 40.0) < 1e-6           # same area despite 5x duration
        assert long.mean() < 0.5 * short.mean()         # per-step shrinks with duration

        basal_short = basal_curve(total_amount=20.0, duration_minutes=780.0)
        basal_long = basal_curve(total_amount=20.0, duration_minutes=1560.0)
        assert abs(basal_short.sum() - 20.0) < 1e-6
        assert abs(basal_long.sum() - 20.0) < 1e-6
        assert basal_long.mean() < basal_short.mean()


class TestSpecCurveTable:
    """SPEC/invariants.md §5 pins these numbers; T1DMDROID's golden vectors are generated from them."""

    def test_gi_gamma_params(self):
        from simulator import gi_gamma_params
        assert gi_gamma_params(100.0) == (2.0, 15.0, 120.0)
        assert gi_gamma_params(50.0) == (3.25, 22.5, 292.5)
        assert gi_gamma_params(-5.0) == gi_gamma_params(0.0)

    def test_bolus_pk_at_15_units(self):
        from simulator import bolus_pk_for_dose, BOLUS_VARIANTS
        x = np.sqrt(15.0) - np.sqrt(5.0)
        for name, theta15, dia15 in [("aspart", 45.0 * (1 + 0.17 * x), 5.6 + 0.8 * x),
                                     ("faster_aspart", 52.0 * (1 + 0.17 * x), 4.7 + 0.8 * x)]:
            v = BOLUS_VARIANTS[name]
            k, theta, dur = bolus_pk_for_dose(15.0, v["gamma_k"], v["gamma_theta"], v["dia_base_hours"])
            assert k == v["gamma_k"]
            assert abs(theta - theta15) < 1e-9
            assert abs(dur - dia15 * 60.0) < 1e-9

    def test_basal_window_leaves_three_percent(self):
        from simulator import BASAL_VARIANTS
        for name, v in BASAL_VARIANTS.items():
            ka, ke, w = v["ka"], v["ke"], v["action_hours"]
            left = (np.exp(-ke * w) / ke - np.exp(-ka * w) / ka) / (1 / ke - 1 / ka)
            assert 0.02 < left <= 0.031, f"{name}: {left:.4f} of the area lies past {w} h"

    def test_basal_tail_reaches_zero(self):
        from simulator import BASAL_VARIANTS
        v = BASAL_VARIANTS["degludec"]
        curve = basal_curve(1.0, v["action_hours"] * 60, v["ka"], v["ke"])
        assert curve[-1] == 0.0 and curve[-2] > 0.0
