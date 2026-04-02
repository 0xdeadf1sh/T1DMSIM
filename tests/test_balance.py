"""
Tests for BG balance under ideal dosing conditions.

Verifies that a perfectly dosed patient (exact ICR match, no counting error,
correct basal that exactly covers HGO) produces near-zero BG delta over time.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulator import (
    T1DMSimulator, gamma_curve, basal_curve,
    HGO_BASE_GRAMS_PER_HOUR, DT_MINUTES,
    BOLUS_GAMMA_K, BOLUS_GAMMA_THETA, BOLUS_DURATION_HOURS,
    BASAL_DURATION_HOURS
)


class TestPerfectBalance:
    def test_hgo_basal_balance(self):
        """A basal dose that exactly covers HGO produces near-zero BG delta.

        With ideal_basal = (HGO_rate * 24h) / ICR, the basal insulin should
        exactly counteract hepatic glucose output, leaving delta ~ 0.
        """
        sim = T1DMSimulator(seed=0, initial_bg=100.0)
        p = sim.patient

        # Manually inject a perfect basal that exactly covers HGO for 24h
        ideal_basal = HGO_BASE_GRAMS_PER_HOUR * 24.0 / p.icr
        duration_min = BASAL_DURATION_HOURS * 60
        perfect_basal = basal_curve(ideal_basal, duration_min)

        # Clear any pending events (no meals, no other doses today)
        sim._pending_events = []
        sim.state.active_curves = []
        sim.state.is_sick = False

        sim.inject_curve(perfect_basal, 0, 'insulin', 'Perfect basal')

        # Run for 12h (avoiding CGM correction behavior by staying near target)
        deltas = []
        for _ in range(12 * 60 // DT_MINUTES):
            step = sim.generate()
            deltas.append(step['bg_delta'])

        mean_delta = np.mean(deltas)
        # With IS noise and HGO noise, expect near-zero mean delta
        # Tolerance: ±5 mg/dL/step average is generous but tests the balance
        assert abs(mean_delta) < 5.0, (
            f"Mean BG delta {mean_delta:.3f} mg/dL/step is too large; "
            "basal should approximately balance HGO")

    def test_meal_bolus_balance(self):
        """A bolus that exactly covers a meal, combined with a basal that covers HGO,
        produces near-zero total BG change over time.

        The test injects:
        - A perfect basal to counteract HGO throughout the window
        - A meal + exact bolus pair (carbs / ICR units)
        So the only sources of BG change are IS noise and physiological guardrails.
        """
        sim = T1DMSimulator(seed=1, initial_bg=100.0)
        p = sim.patient

        # Clear events
        sim._pending_events = []
        sim.state.active_curves = []
        sim.state.is_sick = False

        test_duration_min = 6 * 60  # 6 hours
        meal_grams = 60.0
        bolus_units = meal_grams / p.icr

        meal_curve = gamma_curve(meal_grams, k=3.0, theta=20.0, duration_minutes=300.0)
        bolus_curve = gamma_curve(bolus_units, BOLUS_GAMMA_K, BOLUS_GAMMA_THETA,
                                  BOLUS_DURATION_HOURS * 60)

        # Perfect basal: covers HGO exactly for the test duration
        ideal_basal_rate = HGO_BASE_GRAMS_PER_HOUR / p.icr  # units per hour
        perfect_basal = basal_curve(ideal_basal_rate * (test_duration_min / 60.0), test_duration_min)

        sim.inject_curve(perfect_basal, 0, 'insulin', 'Perfect basal')
        sim.inject_curve(meal_curve, 0, 'carb', 'Test meal')
        sim.inject_curve(bolus_curve, 0, 'insulin', 'Test bolus')

        deltas = []
        for _ in range(test_duration_min // DT_MINUTES):
            step = sim.generate()
            deltas.append(step['bg_delta'])

        # Total BG change should be near zero (±80 mg/dL tolerance for IS noise + curve timing mismatch)
        total_bg_change = sum(deltas)
        assert abs(total_bg_change) < 80.0, (
            f"Total BG change from balanced meal+bolus+basal: {total_bg_change:.1f} mg/dL; "
            "expected near zero (±80 mg/dL tolerance)")

    def test_basal_dose_proportional_to_icr(self):
        """Patients with higher ICR should have lower basal doses (they need less insulin).

        This verifies that basal is tied to HGO/ICR rather than being independent.
        """
        # Find patients with high and low ICR across seeds
        high_icr_patients = []
        low_icr_patients = []

        for seed in range(100):
            sim = T1DMSimulator(seed=seed)
            p = sim.patient
            if p.icr > 12.0:
                high_icr_patients.append(p)
            elif p.icr < 8.0:
                low_icr_patients.append(p)

        if high_icr_patients and low_icr_patients:
            mean_basal_high_icr = np.mean([p.basal_dose for p in high_icr_patients])
            mean_basal_low_icr = np.mean([p.basal_dose for p in low_icr_patients])
            assert mean_basal_high_icr < mean_basal_low_icr, (
                f"High ICR patients (mean basal={mean_basal_high_icr:.1f}U) should need "
                f"less basal than low ICR patients (mean basal={mean_basal_low_icr:.1f}U)")
