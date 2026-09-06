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
    # Glucose-effectiveness / OU equilibrium is an always-on mean-reversion pull
    # toward a stochastic equilibrium — orthogonal to the dose math under test
    # here, and it would drag BG toward the anchor regardless of meal/bolus
    # balance. GE_RATE=0 with GE_RATE_MIN=0 makes each patient's Sg exactly 0.
    monkeypatch.setattr(simulator, 'GE_RATE', 0.0)
    monkeypatch.setattr(simulator, 'GE_RATE_MIN', 0.0)
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
    # Guardrails + corrections would mask a mis-sized dose: renal clearance pulls
    # BG down above 180, counter-regulation/glucagon push it up below 70/55, and
    # the live _check_and_correct rescue loop injects its own boluses. Disabling
    # all four leaves the measured bg_delta as the pure basal-vs-HGO / bolus-vs-
    # meal flux, so a broken dose actually shows up (a zero basal used to pass).
    monkeypatch.setattr(simulator, 'RENAL_CLEARANCE_RATE', 0.0)
    monkeypatch.setattr(simulator, 'COUNTER_REGULATORY_RATE', 0.0)
    monkeypatch.setattr(simulator, 'SEVERE_HYPO_GLUCAGON_RATE', 0.0)
    monkeypatch.setattr(simulator, 'BG_DEATH_MGDL', -1e9)
    monkeypatch.setattr(T1DMSimulator, '_check_and_correct', lambda self, idx: None)
    # Flatten the diurnal-IS profile and daily drift so is_val == is_base across
    # the window; with is_base pinned to 1.0 in _quiet_sim, runtime IS == 1 and
    # the HGO/ICR-sized dose exactly balances instead of drifting on residual IS.
    monkeypatch.setattr(simulator, 'IS_MORNING_AMPLITUDE', 0.0)
    monkeypatch.setattr(simulator, 'IS_NIGHT_DIP_AMPLITUDE', 0.0)
    monkeypatch.setattr(simulator, 'IS_DAILY_DRIFT_SIGMA', 0.0)


def _quiet_sim(seed: int, initial_bg: float = 100.0) -> T1DMSimulator:
    """Clear pending behaviors so injected curves are the only inputs.

    Pins body_weight_kg=75, insulin_resistance_factor=1.0, and is_base=1.0 on
    the generated patient so the HGO scaling matches the test's HGO_BASE
    assumption and (with the diurnal-IS flattening in isolated_biology) runtime
    IS == 1, so the HGO/ICR-sized dose exactly balances. Otherwise a heavy, IR,
    or IS-atypical patient would drift and the balance assertion would not hold.
    """
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
        """A Bateman basal sized to cancel HGO yields ~zero BG delta over its
        broad-peak window — and the test is sensitive to a broken basal.

        isolated_biology disables corrections and the renal/counter-regulatory
        guardrails, so the measured per-step bg_delta is the pure basal-vs-HGO
        flux. A zero basal is therefore no longer masked (it used to pass with a
        final BG near the 400 clamp) and produces a clearly positive drift.
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
        # Sensitivity guard: with the balance removed the flux must drift clearly
        # positive (HGO uncancelled). This is the case that silently passed before.
        assert no_basal - balanced > 1.0, (
            f"A zero basal (mean delta {no_basal:.3f}) must drift far above the "
            f"balanced case ({balanced:.3f}); the test is otherwise insensitive to "
            "the basal dose")

    def test_meal_bolus_balance(self, isolated_biology):
        """A dose-matched bolus cancels a meal (near-zero net BG), and the test
        is sensitive to a mis-dosed bolus.

        With corrections and guardrails disabled by isolated_biology, an over- or
        under-bolus is no longer masked by a rescue carb/bolus or by renal
        clearance: a 2x bolus drives BG sharply down, a 0.5x bolus sharply up.
        The bolus PK window is fixed to the matched dose so all three scales are
        measured over the same horizon.
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
        # Sensitivity guards: a mis-dosed bolus must move BG well off the matched
        # baseline (previously a 2x over-bolus passed via the ±80 tolerance).
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
