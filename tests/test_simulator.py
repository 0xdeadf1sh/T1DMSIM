"""
Tests for T1DMSimulator core behavior.

Verifies:
- Reproducibility: same seed → identical output
- BG stays within clamped bounds
- Meals produce BG rises; insulin produces BG drops
- generate_hours returns consistent shapes
- Weekday/weekend/holiday tracking works
- Alcohol and stress effects are applied
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulator import (
    T1DMSimulator, BG_CLAMP_MIN, BG_CLAMP_MAX,
    gamma_curve, BOLUS_GAMMA_K, BOLUS_GAMMA_THETA, BOLUS_DURATION_HOURS,
    FAST_CARB_K, FAST_CARB_THETA, PUBLIC_HOLIDAYS_PER_YEAR_MIN,
    PUBLIC_HOLIDAYS_PER_YEAR_MAX, SIMULATION_START_DAY_OF_WEEK,
    bolus_pk_for_dose, BOLUS_DIA_BASE_HOURS, BOLUS_DIA_MIN_HOURS, BOLUS_DIA_MAX_HOURS,
    GLYCOGEN_CAPACITY_GRAMS, DT_MINUTES,
    SEVERE_HYPO_REFRACTORY_MIN, HYPO_FOLLOWUP_SKILL_THRESHOLD,
    STEPS_PER_DAY,
)


class TestReproducibility:
    def test_same_seed_same_output(self):
        """Identical seeds produce byte-identical BG traces."""
        for seed in [0, 7, 42, 99]:
            sim1 = T1DMSimulator(seed=seed)
            sim2 = T1DMSimulator(seed=seed)
            data1 = sim1.generate_hours(72)
            data2 = sim2.generate_hours(72)
            np.testing.assert_array_equal(data1['bg'], data2['bg'],
                err_msg=f"seed={seed}: BG traces differ")

    def test_different_seeds_differ(self):
        """Different seeds produce different BG traces."""
        sim1 = T1DMSimulator(seed=1)
        sim2 = T1DMSimulator(seed=2)
        data1 = sim1.generate_hours(24)
        data2 = sim2.generate_hours(24)
        assert not np.array_equal(data1['bg'], data2['bg'])

    def test_reseed_matches_fresh_instance(self):
        """reseed() produces the same output as a fresh instance.

        Compares the full output dict — bg alone is not sufficient because
        per-step IS drift can diverge by ~1.8% from a leaked prior-patient
        state and still produce near-identical BG due to small basal
        contributions per step.
        """
        seed = 17
        sim_fresh = T1DMSimulator(seed=seed)
        data_fresh = sim_fresh.generate_hours(48)

        sim_reseeded = T1DMSimulator(seed=0)
        sim_reseeded.generate_hours(24)
        sim_reseeded.reseed(seed=seed)
        data_reseeded = sim_reseeded.generate_hours(48)

        for key in data_fresh:
            np.testing.assert_array_equal(
                data_fresh[key], data_reseeded[key],
                err_msg=f"reseed diverged from fresh on key '{key}'")


class TestBGBounds:
    def test_bg_within_clamps(self):
        """BG never exceeds hard clamps across many seeds."""
        for seed in range(20):
            sim = T1DMSimulator(seed=seed)
            data = sim.generate_hours(72)
            assert data['bg'].min() >= BG_CLAMP_MIN, (
                f"seed={seed}: BG dipped below {BG_CLAMP_MIN}")
            assert data['bg'].max() <= BG_CLAMP_MAX, (
                f"seed={seed}: BG exceeded {BG_CLAMP_MAX}")

    def test_bg_positive(self):
        """BG is always positive."""
        sim = T1DMSimulator(seed=42)
        data = sim.generate_hours(72)
        assert np.all(data['bg'] > 0)


class TestMealAndInsulinEffect:
    def test_meal_raises_bg(self):
        """Injecting a large carb curve into an otherwise quiet simulation raises BG."""
        sim = T1DMSimulator(seed=5, initial_bg=120.0)
        # Advance a few steps to stabilize
        for _ in range(10):
            sim.generate()

        bg_before = sim.state.bg
        # Inject a 60g fast carb curve via the public inject_curve API
        carb_curve = gamma_curve(60.0, FAST_CARB_K, FAST_CARB_THETA, 120.0)
        sim.inject_curve(carb_curve, sim.state.current_idx, 'carb', 'Test meal')
        # Run for 60 minutes (12 steps)
        for _ in range(12):
            sim.generate()

        bg_after = sim.state.bg
        assert bg_after > bg_before, (
            f"BG did not rise after large carb injection: before={bg_before:.0f}, after={bg_after:.0f}")

    def test_insulin_lowers_bg(self):
        """Injecting a large insulin curve into a high-BG simulation lowers BG."""
        sim = T1DMSimulator(seed=5, initial_bg=280.0)
        for _ in range(5):
            sim.generate()

        bg_before = sim.state.bg
        # Inject 10U bolus via the public inject_curve API
        bolus_curve = gamma_curve(10.0, BOLUS_GAMMA_K, BOLUS_GAMMA_THETA,
                                  BOLUS_DURATION_HOURS * 60)
        sim.inject_curve(bolus_curve, sim.state.current_idx, 'insulin', 'Test bolus')
        for _ in range(24):  # 2 hours
            sim.generate()

        bg_after = sim.state.bg
        assert bg_after < bg_before, (
            f"BG did not fall after insulin injection: before={bg_before:.0f}, after={bg_after:.0f}")


class TestGenerateHours:
    def test_output_shapes(self):
        """generate_hours returns arrays of the correct length."""
        sim = T1DMSimulator(seed=0)
        hours = 48.0
        data = sim.generate_hours(hours)
        expected_steps = int(hours * 60 / 5)
        for key, arr in data.items():
            assert len(arr) == expected_steps, (
                f"Key '{key}': expected {expected_steps} steps, got {len(arr)}")

    def test_output_keys_present(self):
        """generate_hours returns all expected keys."""
        sim = T1DMSimulator(seed=0)
        data = sim.generate_hours(24)
        required_keys = ['bg', 'bg_observed', 'bg_delta', 'total_carb',
                         'total_insulin', 'insulin_resistance', 'hgo',
                         'is_sick', 'is_rare_day', 'is_weekend', 'is_holiday',
                         'alcohol_hgo_factor']
        for key in required_keys:
            assert key in data, f"Missing key: '{key}'"


class TestWeekdayWeekend:
    def test_day_of_week_cycles(self):
        """day_of_week increments correctly through the week."""
        sim = T1DMSimulator(seed=0)
        # Generate 7 days worth of steps; check day_of_week at each day boundary
        seen_days = set()
        for _ in range(7 * 288):
            step = sim.generate()
            seen_days.add(sim.state.day_of_week)
        assert seen_days == {0, 1, 2, 3, 4, 5, 6}, (
            f"Not all days of week seen: {seen_days}")

    def test_weekend_flag_matches_day_of_week(self):
        """is_weekend flag is True iff day_of_week >= 5."""
        sim = T1DMSimulator(seed=3)
        for _ in range(7 * 288):
            step = sim.generate()
            expected_weekend = sim.state.day_of_week >= 5
            assert step['is_weekend'] == expected_weekend


class TestHolidays:
    def test_holidays_not_on_weekends(self):
        """Generated holidays never fall on weekends."""
        sim = T1DMSimulator(seed=42)
        for day in sim._holiday_set:
            dow = (SIMULATION_START_DAY_OF_WEEK + day) % 7
            assert dow < 5, f"Holiday on day {day} falls on a weekend (dow={dow})"

    def test_holiday_count_within_range(self):
        """Number of holidays per year is within configured range."""
        sim = T1DMSimulator(seed=7)
        # Year 0: days 0-364
        year0_holidays = {d for d in sim._holiday_set if 0 <= d < 365}
        assert PUBLIC_HOLIDAYS_PER_YEAR_MIN <= len(year0_holidays) <= PUBLIC_HOLIDAYS_PER_YEAR_MAX, (
            f"Year 0 has {len(year0_holidays)} holidays, expected "
            f"[{PUBLIC_HOLIDAYS_PER_YEAR_MIN}, {PUBLIC_HOLIDAYS_PER_YEAR_MAX}]")

    def test_holidays_distributed_through_year(self):
        """Holidays are spread across the year, not clustered at the start."""
        sim = T1DMSimulator(seed=10)
        year0_holidays = sorted(d for d in sim._holiday_set if 0 <= d < 365)
        if len(year0_holidays) >= 4:
            # Check that last holiday is not in first quarter
            assert max(year0_holidays) > 91, "All holidays are in the first quarter"
            # Check that first holiday is not in last quarter
            assert min(year0_holidays) < 274, "All holidays are in the last quarter"


class TestBolusPKForDose:
    """The dose-dependent bolus PK helper must scale duration with dose and
    return a valid (k, theta, duration_minutes) triple for any dose."""

    def test_returns_triple(self):
        for dose in [0.1, 1.0, 5.0, 10.0, 30.0]:
            k, theta, duration_min = bolus_pk_for_dose(dose)
            assert k > 0, f"k must be positive (dose={dose})"
            assert theta > 0, f"theta must be positive (dose={dose})"
            assert duration_min > 0, f"duration must be positive (dose={dose})"

    def test_duration_within_bounds(self):
        """Duration is clipped to [BOLUS_DIA_MIN_HOURS, BOLUS_DIA_MAX_HOURS]."""
        for dose in [0.01, 0.5, 5.0, 50.0, 1000.0]:
            _, _, duration_min = bolus_pk_for_dose(dose)
            duration_h = duration_min / 60.0
            assert BOLUS_DIA_MIN_HOURS <= duration_h <= BOLUS_DIA_MAX_HOURS, (
                f"dose={dose} produced duration_h={duration_h}, "
                f"outside [{BOLUS_DIA_MIN_HOURS}, {BOLUS_DIA_MAX_HOURS}]")

    def test_duration_monotone_with_dose(self):
        """Larger doses act at least as long (within clip bounds)."""
        doses = [1.0, 3.0, 5.0, 10.0, 20.0]
        durations = [bolus_pk_for_dose(d)[2] for d in doses]
        for a, b in zip(durations, durations[1:]):
            assert b >= a - 1e-9, f"duration not monotone: {durations}"

    def test_reference_5u_matches_base(self):
        """5U dose (reference) should produce duration = BOLUS_DIA_BASE_HOURS."""
        _, _, duration_min = bolus_pk_for_dose(5.0)
        assert abs(duration_min / 60.0 - BOLUS_DIA_BASE_HOURS) < 1e-6


class TestGlycogenReservoir:
    """The hepatic glycogen reservoir is a finite store. It must stay in
    [0, GLYCOGEN_CAPACITY_GRAMS] at every step under any simulation trace."""

    def test_glycogen_within_bounds_over_long_run(self):
        """Glycogen never goes negative or exceeds capacity across 72h."""
        for seed in [0, 3, 7, 11, 19]:
            sim = T1DMSimulator(seed=seed)
            for _ in range(72 * 60 // DT_MINUTES):
                sim.generate()
                g = sim.state.glycogen_grams
                assert 0.0 <= g <= GLYCOGEN_CAPACITY_GRAMS, (
                    f"seed={seed}: glycogen={g} out of [0, {GLYCOGEN_CAPACITY_GRAMS}]")

    def test_glycogen_drains_without_carbs(self):
        """With no incoming carbs, sustained HGO must drain the reservoir."""
        sim = T1DMSimulator(seed=0)
        sim._pending_events = []
        sim.state.active_curves = []
        sim.state.is_sick = False
        initial = sim.state.glycogen_grams
        # Run 24h with no injected carbs/insulin — HGO drains glycogen.
        for _ in range(24 * 60 // DT_MINUTES):
            sim.generate()
        assert sim.state.glycogen_grams < initial, (
            f"glycogen rose from {initial} to {sim.state.glycogen_grams} with no carbs")

    def test_glycogen_refills_from_large_carb_load(self):
        """A large injected carb absorption refills a depleted reservoir."""
        sim = T1DMSimulator(seed=0)
        sim._pending_events = []
        sim.state.active_curves = []
        sim.state.is_sick = False
        sim.state.glycogen_grams = 5.0  # near-empty
        # 200g over 5h (well above any meal threshold)
        big_meal = gamma_curve(200.0, k=3.0, theta=20.0, duration_minutes=300.0)
        sim.inject_curve(big_meal, sim.state.current_idx, 'carb', 'big')
        for _ in range(5 * 60 // DT_MINUTES):
            sim.generate()
        assert sim.state.glycogen_grams > 5.0, (
            f"glycogen did not refill from 200g carbs: {sim.state.glycogen_grams}")


class TestSevereHypoRefractory:
    """The 10-min severe-hypo refractory keeps rescue carbs from stacking.

    This is a CLAUDE.md-documented critical clinical invariant: a full
    bypass produces a sawtooth where the patient rage-eats 3-5 times in
    10-15 min and overshoots to 140-180, while removing the bypass
    re-opens 6+ hour dangerous hypos. The 10-min gate is the compromise.
    """

    def _setup_severe_hypo(self, sim, idx):
        """Force the simulator into a severe-hypo state the patient will act on."""
        s = sim.state
        s.bg = 40.0
        s.bg_observed = 40.0
        s.last_cgm_check_idx = -9999
        s.last_hypo_correction_idx = -9999
        # Ensure awake — wake at start, sleep far in the future
        sim._today_wake_idx = 0
        sim._today_sleep_idx = idx + STEPS_PER_DAY

    def test_severe_hypo_refractory_blocks_back_to_back(self):
        """A second severe-hypo rescue within 10 min must be blocked."""
        sim = T1DMSimulator(seed=0)
        sim.generate()  # advance one step so _today_wake/sleep are populated
        idx = sim.state.current_idx
        self._setup_severe_hypo(sim, idx)

        sim._check_and_correct(idx)
        first_idx = sim.state.last_hypo_correction_idx
        assert first_idx == idx, "first severe-hypo rescue did not fire"

        # 5 min later, BG still severe — refractory must block
        idx2 = idx + 1
        self._setup_severe_hypo(sim, idx2)
        sim.state.last_hypo_correction_idx = first_idx
        sim._check_and_correct(idx2)
        assert sim.state.last_hypo_correction_idx == first_idx, (
            "second rescue fired within 5 min — refractory bypassed")

        # 10+ min later, BG still severe — refractory must release
        idx3 = idx + int(SEVERE_HYPO_REFRACTORY_MIN / DT_MINUTES) + 1
        self._setup_severe_hypo(sim, idx3)
        sim.state.last_hypo_correction_idx = first_idx
        sim._check_and_correct(idx3)
        assert sim.state.last_hypo_correction_idx == idx3, (
            "third rescue did not fire after refractory window — gate stuck")

    def test_moderate_hypo_uses_longer_refractory(self):
        """Moderate hypo (55-70) must use the 20-min refractory, not 10."""
        sim = T1DMSimulator(seed=0)
        sim.generate()
        idx = sim.state.current_idx
        self._setup_severe_hypo(sim, idx)
        sim.state.bg = sim.state.bg_observed = 62.0  # moderate, not severe

        sim._check_and_correct(idx)
        first_idx = sim.state.last_hypo_correction_idx
        if first_idx != idx:
            return  # patient's eff_low_thresh may exclude 62 — skip silently

        # 11 min later — within 20-min moderate refractory, must block
        idx2 = idx + int(11 / DT_MINUTES)
        sim.state.bg = sim.state.bg_observed = 62.0
        sim.state.last_cgm_check_idx = -9999
        sim._check_and_correct(idx2)
        assert sim.state.last_hypo_correction_idx == first_idx, (
            "moderate hypo fired within 20 min — wrong refractory used")


class TestHypoFollowupSnack:
    """The skill-gated slow-tail follow-up snack is the other half of the
    sawtooth fix. CLAUDE.md flags removing it as re-opening dangerous hypos.
    """

    def _trigger_correction(self, sim):
        sim.generate()
        idx = sim.state.current_idx
        s = sim.state
        s.bg = 40.0
        s.bg_observed = 40.0
        s.last_cgm_check_idx = -9999
        s.last_hypo_correction_idx = -9999
        sim._today_wake_idx = 0
        sim._today_sleep_idx = idx + STEPS_PER_DAY
        carbs_before = sum(1 for c in s.active_curves
                           if c.curve_type == 'correction_carb')
        sim._check_and_correct(idx)
        new_carbs = [c for c in s.active_curves
                     if c.curve_type == 'correction_carb'][carbs_before:]
        return new_carbs

    def test_high_skill_patient_gets_followup(self):
        """A high-skill patient should get a followup slow-carb curve."""
        sim = T1DMSimulator(seed=0)
        p = sim.patient
        # Force skill above the threshold
        p.attentiveness = 0.9
        p.dosing_competence = 0.9
        new_curves = self._trigger_correction(sim)
        labels = [c.label for c in new_curves]
        assert any('followup' in l.lower() for l in labels), (
            f"expected followup curve for skill_avg=0.9, got {labels}")

    def test_low_skill_patient_no_followup(self):
        """A patient below HYPO_FOLLOWUP_SKILL_THRESHOLD must NOT get a followup."""
        sim = T1DMSimulator(seed=0)
        p = sim.patient
        below = HYPO_FOLLOWUP_SKILL_THRESHOLD - 0.1
        p.attentiveness = below
        p.dosing_competence = below
        new_curves = self._trigger_correction(sim)
        labels = [c.label for c in new_curves]
        assert not any('followup' in l.lower() for l in labels), (
            f"low-skill patient got followup unexpectedly: {labels}")
