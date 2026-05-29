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
    BASAL_DURATION_HOURS,
)


# Measurement window for a single Bateman basal_curve. The curve rises to a
# broad peak around 6h post-injection (tmax = ln(ka/ke)/(ka-ke) ≈ 6.3h for the
# default ka=0.30, ke=0.07), so 4h warmup lands inside the rising phase and 8h
# of measurement straddles the peak — the broadest, flattest region of the
# unit curve and the natural place to enforce HGO/basal balance.
_BAL_WARMUP_HOURS = 4.0
_BAL_MEASURE_HOURS = 8.0


def _balanced_basal_total(unit_curve: np.ndarray, hgo_per_step_units: float,
                          warmup_steps: int, measure_steps: int) -> float:
    """Total basal dose such that the mean per-step delivery across the
    measurement window equals ``hgo_per_step_units``.

    basal_curve is normalized so sum(values)=total_amount. The per-step rate
    varies across a Bateman curve, so "perfect balance" can only be enforced
    on average over a finite window. Scaling the unit curve by
    ``hgo_per_step_units / mean(unit_curve[window])`` produces a dose whose
    mean delivery in the window equals the HGO-cancelling rate.
    """
    window_mean_unit = float(np.mean(unit_curve[warmup_steps:warmup_steps + measure_steps]))
    return hgo_per_step_units / window_mean_unit


@pytest.fixture
def isolated_biology(monkeypatch):
    """Disable stochastic / feedback biology that intentionally breaks the
    'exact-ICR-match = zero delta' assumption these tests rely on:

    - postprandial insulin resistance (IR rises while carbs absorb -> bolus under-corrects)
    - lipohypertrophy per-dose multiplier (insulin amplitude noise)
    - glucotoxicity (IS drift from chronic hyperglycemia)
    - per-step absorption noise on carbs and insulin

    With these off, dose math is the only thing under test.
    """
    monkeypatch.setattr(simulator, 'POSTPRANDIAL_IR_PENALTY_FACTOR', 0.0)
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
    """Clear pending behaviors so injected curves are the only inputs.

    Pins body_weight_kg=75 and insulin_resistance_factor=1.0 on the generated
    patient so the HGO scaling matches the test's HGO_BASE assumption.
    Otherwise a heavy or IR patient would have hgo_value scaled and the
    plateau-basal-vs-HGO balance assertion would no longer hold.
    """
    sim = T1DMSimulator(seed=seed, initial_bg=initial_bg)
    sim._pending_events = []
    sim.state.active_curves = []
    sim.state.is_sick = False
    sim.patient.body_weight_kg = 75.0
    sim.patient.insulin_resistance_factor = 1.0
    return sim


class TestPerfectBalance:
    def test_hgo_basal_balance(self, isolated_biology):
        """A Bateman basal sized for its broad-peak window yields ~zero BG delta.

        basal_curve is a Bateman one-compartment PK (smooth rise to a broad
        peak at ~6.3h, exponential decline). Per-step insulin delivery varies
        across the curve, so "perfect balance" can only be enforced on average
        over a finite window. We size the dose so the mean insulin delivery
        across the measurement window equals the HGO-cancelling rate, warm up
        past the rising phase, then measure across the broad-peak window.
        """
        sim = _quiet_sim(seed=0)
        p = sim.patient

        duration_min = BASAL_DURATION_HOURS * 60
        warmup_steps = int(_BAL_WARMUP_HOURS * 60) // DT_MINUTES
        measure_steps = int(_BAL_MEASURE_HOURS * 60) // DT_MINUTES

        unit_curve = basal_curve(1.0, duration_min)
        hgo_per_step_units = HGO_BASE_GRAMS_PER_HOUR * (DT_MINUTES / 60.0) / p.icr
        ideal_total = _balanced_basal_total(
            unit_curve, hgo_per_step_units, warmup_steps, measure_steps)

        perfect_basal = basal_curve(ideal_total, duration_min)
        sim.inject_curve(perfect_basal, 0, 'insulin', 'Perfect basal')

        for _ in range(warmup_steps):
            sim.generate()

        deltas = []
        for _ in range(measure_steps):
            deltas.append(sim.generate()['bg_delta'])

        mean_delta = float(np.mean(deltas))
        # Plasma-insulin EMA introduces phase lag against the rising/falling
        # Bateman curve so per-step deltas don't cancel exactly; ±2.5 mg/dL/step
        # over 8h still catches a broken balance (the previous trapezoidal
        # version delivered ~0.06 U total — orders of magnitude below the
        # HGO-cancelling rate — and silently passed via the BG ceiling clamp).
        assert abs(mean_delta) < 2.5, (
            f"Mean BG delta {mean_delta:.3f} mg/dL/step exceeds 2.5; "
            "Bateman basal sized for the broad-peak window should approximately balance HGO")

    def test_meal_bolus_balance(self, isolated_biology):
        """Balanced meal + dose-matched bolus + balanced basal → near-zero net BG.

        We size the Bateman basal so its broad-peak window cancels HGO, warm up
        past the rising phase, then inject a meal and a bolus computed via
        bolus_pk_for_dose (the dose-dependent PK helper, not the legacy fixed-
        duration constants). We measure for long enough that both meal
        absorption and bolus action complete inside the window.
        """
        sim = _quiet_sim(seed=1)
        p = sim.patient

        # --- Balanced basal sized as in test_hgo_basal_balance ---
        basal_dur_min = BASAL_DURATION_HOURS * 60
        warmup_steps = int(_BAL_WARMUP_HOURS * 60) // DT_MINUTES
        measure_steps = int(_BAL_MEASURE_HOURS * 60) // DT_MINUTES
        unit_curve = basal_curve(1.0, basal_dur_min)
        hgo_per_step_units = HGO_BASE_GRAMS_PER_HOUR * (DT_MINUTES / 60.0) / p.icr
        ideal_total = _balanced_basal_total(
            unit_curve, hgo_per_step_units, warmup_steps, measure_steps)
        perfect_basal = basal_curve(ideal_total, basal_dur_min)
        sim.inject_curve(perfect_basal, 0, 'insulin', 'Perfect basal')

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
