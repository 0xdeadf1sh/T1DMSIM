"""Tests for BG balance under ideal dosing conditions (perfect ICR/basal -> ~zero delta)."""

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


# Basal tmax ≈ 6.3h (ka=0.30, ke=0.07); 4h warmup + 8h measurement straddles that peak.
_BAL_WARMUP_HOURS = 4.0
_BAL_MEASURE_HOURS = 8.0


def _balanced_basal_total(unit_curve: np.ndarray, hgo_per_step_units: float,
                          warmup_steps: int, measure_steps: int) -> float:
    """Total basal dose whose mean per-step delivery in the window equals hgo_per_step_units."""
    window_mean_unit = float(np.mean(unit_curve[warmup_steps:warmup_steps + measure_steps]))
    return hgo_per_step_units / window_mean_unit


@pytest.fixture
def isolated_biology(monkeypatch):
    """Disable stochastic/feedback biology (postprandial IR, lipohypertrophy, glucotoxicity,
    absorption noise) that breaks the 'exact-ICR-match = zero delta' assumption, so dose
    math is the only thing under test.
    """
    monkeypatch.setattr(simulator, 'POSTPRANDIAL_IR_PENALTY_FACTOR', 0.0)
    monkeypatch.setattr(simulator, 'SITE_QUALITY_SIGMA_BASE', 0.0)
    monkeypatch.setattr(simulator, 'GLUCOTOX_MAX_IS_INCREASE', 0.0)
    # GE_RATE=0 with GE_RATE_MIN=0 zeroes Sg: the OU pull would drag BG off the balance test.
    monkeypatch.setattr(simulator, 'GE_RATE', 0.0)
    monkeypatch.setattr(simulator, 'GE_RATE_MIN', 0.0)
    monkeypatch.setattr(simulator, 'CARB_ABSORPTION_NOISE_SIGMA', 0.0)
    monkeypatch.setattr(simulator, 'INSULIN_ABSORPTION_NOISE_SIGMA', 0.0)
    # Circadian HGO straddles the ~6.5am dawn peak, skewing glucose_in +30-40% otherwise.
    monkeypatch.setattr(simulator, 'DAWN_HGO_AMPLITUDE_MEAN', 0.0)
    monkeypatch.setattr(simulator, 'DAWN_HGO_AMPLITUDE_SIGMA', 0.0)
    monkeypatch.setattr(simulator, 'NIGHT_HGO_DIP_AMPLITUDE_MEAN', 0.0)
    monkeypatch.setattr(simulator, 'NIGHT_HGO_DIP_AMPLITUDE_SIGMA', 0.0)
    # Flatten HGO to HGO_BASE; else the bolus's insulin spike over-suppresses it via the Hill fn.
    monkeypatch.setattr(simulator, 'HGO_UNSUPPRESSED_GRAMS_PER_HOUR',
                        simulator.HGO_BASE_GRAMS_PER_HOUR)
    monkeypatch.setattr(simulator, 'HGO_SUPPRESSED_FLOOR_GRAMS_PER_HOUR',
                        simulator.HGO_BASE_GRAMS_PER_HOUR)
    # Disabling all four leaves bg_delta as pure dose flux, so a broken dose shows up.
    monkeypatch.setattr(simulator, 'RENAL_CLEARANCE_RATE', 0.0)
    monkeypatch.setattr(simulator, 'COUNTER_REGULATORY_RATE', 0.0)
    monkeypatch.setattr(simulator, 'SEVERE_HYPO_GLUCAGON_RATE', 0.0)
    monkeypatch.setattr(T1DMSimulator, '_check_and_correct', lambda self, idx: None)
    # Flatten diurnal-IS + drift so is_val == is_base == 1.0, else the sized dose drifts.
    monkeypatch.setattr(simulator, 'IS_MORNING_AMPLITUDE', 0.0)
    monkeypatch.setattr(simulator, 'IS_NIGHT_DIP_AMPLITUDE', 0.0)
    monkeypatch.setattr(simulator, 'IS_DAILY_DRIFT_SIGMA', 0.0)


def _quiet_sim(seed: int, initial_bg: float = 100.0) -> T1DMSimulator:
    """Clear pending behaviors and pin weight/IR/is_base to 1.0 so the sized dose balances."""
    sim = T1DMSimulator(seed=seed, initial_bg=initial_bg)
    sim._pending_events = []
    sim.state.active_curves = []
    sim.state.is_sick = False
    sim.patient.body_weight_kg = 75.0
    sim.patient.insulin_resistance_factor = 1.0
    sim.patient.is_base = 1.0
    return sim


class TestPerfectBalance:
    def test_hgo_basal_balance(self, isolated_biology):
        """A Bateman basal sized to cancel HGO yields ~zero BG delta over its broad-peak
        window; with guardrails disabled a zero basal now drifts clearly positive instead.
        """
        duration_min = BASAL_DURATION_HOURS * 60
        warmup_steps = int(_BAL_WARMUP_HOURS * 60) // DT_MINUTES
        measure_steps = int(_BAL_MEASURE_HOURS * 60) // DT_MINUTES

        def mean_delta(total: float) -> float:
            sim = _quiet_sim(seed=0)
            if total > 0:
                sim.inject_curve(basal_curve(total, duration_min), 0, 'insulin', 'basal')
            for _ in range(warmup_steps):
                sim.generate()
            return float(np.mean([sim.generate()['bg_delta'] for _ in range(measure_steps)]))

        icr = _quiet_sim(seed=0).patient.icr
        unit_curve = basal_curve(1.0, duration_min)
        hgo_per_step_units = HGO_BASE_GRAMS_PER_HOUR * (DT_MINUTES / 60.0) / icr
        ideal_total = _balanced_basal_total(
            unit_curve, hgo_per_step_units, warmup_steps, measure_steps)

        balanced = mean_delta(ideal_total)
        no_basal = mean_delta(0.0)

        assert abs(balanced) < 0.4, (
            f"Balanced basal mean delta {balanced:.3f} mg/dL/step should be ~0")
        # Sensitivity guard: no-basal drift must be clearly positive (silently passed before).
        assert no_basal - balanced > 1.0, (
            f"A zero basal (mean delta {no_basal:.3f}) must drift far above the "
            f"balanced case ({balanced:.3f}); the test is otherwise insensitive to "
            "the basal dose")

    def test_meal_bolus_balance(self, isolated_biology):
        """A dose-matched bolus cancels a meal; with guardrails disabled, a 2x bolus
        drives BG sharply down and a 0.5x bolus sharply up, over the same PK horizon.
        """
        basal_dur_min = BASAL_DURATION_HOURS * 60
        warmup_steps = int(_BAL_WARMUP_HOURS * 60) // DT_MINUTES
        measure_steps = int(_BAL_MEASURE_HOURS * 60) // DT_MINUTES

        def total_change(bolus_scale: float) -> float:
            sim = _quiet_sim(seed=1)
            p = sim.patient
            unit_curve = basal_curve(1.0, basal_dur_min)
            hgo_per_step_units = HGO_BASE_GRAMS_PER_HOUR * (DT_MINUTES / 60.0) / p.icr
            ideal_total = _balanced_basal_total(
                unit_curve, hgo_per_step_units, warmup_steps, measure_steps)
            sim.inject_curve(basal_curve(ideal_total, basal_dur_min), 0, 'insulin', 'Perfect basal')
            for _ in range(warmup_steps):
                sim.generate()

            meal_grams = 60.0
            matched_units = meal_grams / p.icr
            meal_duration_min = 300.0
            meal = gamma_curve(meal_grams, k=3.0, theta=20.0,
                               duration_minutes=meal_duration_min)
            bk, btheta, bdur_min = bolus_pk_for_dose(matched_units)
            bolus = gamma_curve(matched_units * bolus_scale, bk, btheta, bdur_min)

            cur_idx = sim.state.current_idx
            sim.inject_curve(meal, cur_idx, 'carb', 'Test meal')
            sim.inject_curve(bolus, cur_idx, 'insulin', 'Test bolus')

            window_min = max(meal_duration_min, bdur_min) + 60.0  # +1h tail
            return sum(sim.generate()['bg_delta'] for _ in range(int(window_min) // DT_MINUTES))

        matched = total_change(1.0)
        over = total_change(2.0)
        under = total_change(0.5)

        assert abs(matched) < 30.0, (
            f"Dose-matched meal+bolus net change {matched:.1f} mg/dL should be ~0")
        # Sensitivity guard: a mis-dosed bolus must move BG well off the matched baseline.
        assert matched - over > 50.0, (
            f"A 2x over-bolus ({over:.1f}) must drive BG well below the matched "
            f"case ({matched:.1f})")
        assert under - matched > 50.0, (
            f"A 0.5x under-bolus ({under:.1f}) must drive BG well above the matched "
            f"case ({matched:.1f})")

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
