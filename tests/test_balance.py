"""
Tests for BG balance under ideal dosing conditions.

Verifies that a perfectly dosed patient (exact ICR match, no counting error,
correct basal that exactly covers HGO) produces near-zero BG delta over time.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import simulator
from simulator import (
    T1DMSimulator, gamma_curve, basal_curve, bolus_pk_for_dose,
    HGO_BASE_GRAMS_PER_HOUR, DT_MINUTES,
    BASAL_DURATION_HOURS, BASAL_RAMP_UP_HOURS, BASAL_RAMP_DOWN_HOURS,
)


@pytest.fixture
def isolated_biology(monkeypatch):
    """Disable stochastic / feedback biology that intentionally breaks the
    'exact-ICR-match = zero delta' assumption these tests rely on:

    - postprandial IS bonus (IS rises while carbs absorb -> bolus over-corrects)
    - lipohypertrophy per-dose multiplier (insulin amplitude noise)
    - glucotoxicity (IS drift from chronic hyperglycemia)
    - per-step absorption noise on carbs and insulin

    With these off, dose math is the only thing under test.
    """
    monkeypatch.setattr(simulator, 'POSTPRANDIAL_IS_BONUS_FACTOR', 0.0)
    monkeypatch.setattr(simulator, 'SITE_QUALITY_SIGMA_BASE', 0.0)
    monkeypatch.setattr(simulator, 'GLUCOTOX_MAX_IS_INCREASE', 0.0)
    monkeypatch.setattr(simulator, 'CARB_ABSORPTION_NOISE_SIGMA', 0.0)
    monkeypatch.setattr(simulator, 'INSULIN_ABSORPTION_NOISE_SIGMA', 0.0)
    # Circadian HGO would skew the 8h measurement window (which straddles the
    # dawn peak at ~6.5am) so the basal-vs-HGO balance test would see a
    # systematic +30-40% glucose_in spike during measurement.
    monkeypatch.setattr(simulator, 'DAWN_HGO_AMPLITUDE_MEAN', 0.0)
    monkeypatch.setattr(simulator, 'DAWN_HGO_AMPLITUDE_SIGMA', 0.0)
    monkeypatch.setattr(simulator, 'NIGHT_HGO_DIP_AMPLITUDE_MEAN', 0.0)
    monkeypatch.setattr(simulator, 'NIGHT_HGO_DIP_AMPLITUDE_SIGMA', 0.0)
    # Flatten HGO to a constant = HGO_BASE so basal sized against HGO_BASE
    # exactly balances; otherwise the bolus's plasma-insulin spike over-suppresses
    # HGO via the Hill function and the basal becomes "too much" during meals.
    monkeypatch.setattr(simulator, 'HGO_UNSUPPRESSED_GRAMS_PER_HOUR',
                        simulator.HGO_BASE_GRAMS_PER_HOUR)
    monkeypatch.setattr(simulator, 'HGO_SUPPRESSED_FLOOR_GRAMS_PER_HOUR',
                        simulator.HGO_BASE_GRAMS_PER_HOUR)


def _quiet_sim(seed: int, initial_bg: float = 100.0) -> T1DMSimulator:
    """Clear pending behaviors so injected curves are the only inputs."""
    sim = T1DMSimulator(seed=seed, initial_bg=initial_bg)
    sim._pending_events = []
    sim.state.active_curves = []
    sim.state.is_sick = False
    return sim


class TestPerfectBalance:
    def test_hgo_basal_balance(self, isolated_biology):
        """A basal at the plateau rate that matches HGO yields ~zero BG delta.

        basal_curve produces a trapezoid (ramp-up + plateau + ramp-down) whose
        plateau height = total_dose / sum_of_unit_trapezoid. We size the dose so
        the plateau insulin-per-step exactly cancels HGO-per-step, then warm up
        past ramp-up before measuring deltas. With everything balanced, the
        residual is just noise (IS jitter, glucotox/glycogen second-order
        coupling) — small per step, mean-zero.
        """
        sim = _quiet_sim(seed=0)
        p = sim.patient

        duration_min = BASAL_DURATION_HOURS * 60
        # Reconstruct the trapezoid sum to size the total dose so the plateau
        # value equals the HGO-cancelling insulin per step.
        unit_trap = basal_curve(1.0, duration_min,
                                ramp_up_hours=BASAL_RAMP_UP_HOURS,
                                ramp_down_hours=BASAL_RAMP_DOWN_HOURS)
        trap_sum = float(np.sum(unit_trap))
        hgo_per_step_units = HGO_BASE_GRAMS_PER_HOUR * (DT_MINUTES / 60.0) / p.icr
        ideal_total = hgo_per_step_units * trap_sum

        perfect_basal = basal_curve(ideal_total, duration_min,
                                    ramp_up_hours=BASAL_RAMP_UP_HOURS,
                                    ramp_down_hours=BASAL_RAMP_DOWN_HOURS)
        sim.inject_curve(perfect_basal, 0, 'insulin', 'Perfect basal')

        # Warm past ramp-up + insulin-smoothing lag (~30 min) into the plateau.
        warmup_steps = int((BASAL_RAMP_UP_HOURS + 0.5) * 60) // DT_MINUTES
        for _ in range(warmup_steps):
            sim.generate()

        # Measure across plateau window only — well before ramp-down kicks in.
        deltas = []
        for _ in range(8 * 60 // DT_MINUTES):
            deltas.append(sim.generate()['bg_delta'])

        mean_delta = np.mean(deltas)
        # Residual drift comes from diurnal IS variation and HGO Hill curvature
        # at off-reference ICR; 2.5 mg/dL/step (~240 mg/dL across 96 steps) is
        # tighter than the legacy 5.0 and still catches a broken balance.
        assert abs(mean_delta) < 2.5, (
            f"Mean BG delta {mean_delta:.3f} mg/dL/step exceeds 2.5; "
            "plateau basal should approximately balance HGO")

    def test_meal_bolus_balance(self, isolated_biology):
        """Balanced meal + dose-matched bolus + plateau basal → near-zero net BG.

        We size the basal so its plateau cancels HGO, warm up past ramp-up, then
        inject a meal and a bolus computed via bolus_pk_for_dose (the
        dose-dependent PK helper, not the legacy fixed-duration constants). We
        measure for long enough that both meal absorption and bolus action
        complete inside the window.
        """
        sim = _quiet_sim(seed=1)
        p = sim.patient

        # --- Plateau basal sized as in test_hgo_basal_balance ---
        basal_dur_min = BASAL_DURATION_HOURS * 60
        unit_trap = basal_curve(1.0, basal_dur_min,
                                ramp_up_hours=BASAL_RAMP_UP_HOURS,
                                ramp_down_hours=BASAL_RAMP_DOWN_HOURS)
        hgo_per_step_units = HGO_BASE_GRAMS_PER_HOUR * (DT_MINUTES / 60.0) / p.icr
        ideal_total = hgo_per_step_units * float(np.sum(unit_trap))
        perfect_basal = basal_curve(ideal_total, basal_dur_min,
                                    ramp_up_hours=BASAL_RAMP_UP_HOURS,
                                    ramp_down_hours=BASAL_RAMP_DOWN_HOURS)
        sim.inject_curve(perfect_basal, 0, 'insulin', 'Perfect basal')

        warmup_steps = int((BASAL_RAMP_UP_HOURS + 0.5) * 60) // DT_MINUTES
        for _ in range(warmup_steps):
            sim.generate()

        # --- Meal and dose-matched bolus, injected at the same step ---
        meal_grams = 60.0
        bolus_units = meal_grams / p.icr
        meal_duration_min = 300.0
        meal = gamma_curve(meal_grams, k=3.0, theta=20.0,
                           duration_minutes=meal_duration_min)
        bk, btheta, bdur_min = bolus_pk_for_dose(bolus_units)
        bolus = gamma_curve(bolus_units, bk, btheta, bdur_min)

        cur_idx = sim.state.current_idx
        sim.inject_curve(meal, cur_idx, 'carb', 'Test meal')
        sim.inject_curve(bolus, cur_idx, 'insulin', 'Test bolus')

        # Measure long enough for both curves to fully resolve.
        window_min = max(meal_duration_min, bdur_min) + 60.0  # +1h tail
        deltas = []
        for _ in range(int(window_min) // DT_MINUTES):
            deltas.append(sim.generate()['bg_delta'])

        total_bg_change = sum(deltas)
        # Stochastic biology disabled by isolated_biology fixture; residual
        # drift is diurnal IS variation + curve-timing alignment. ±60 mg/dL
        # total still catches gross dose/carb imbalance and is tighter than
        # the legacy ±80 against a more rigorous setup.
        assert abs(total_bg_change) < 60.0, (
            f"Total BG change from balanced meal+bolus+basal: {total_bg_change:.1f} mg/dL; "
            "expected near zero (±60 mg/dL)")

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
