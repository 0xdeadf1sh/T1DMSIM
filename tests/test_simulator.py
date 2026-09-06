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
import simulator
from simulator import (
    T1DMSimulator, BG_CLAMP_MIN, BG_CLAMP_MAX,
    gamma_curve, BOLUS_GAMMA_K, BOLUS_GAMMA_THETA, BOLUS_DURATION_HOURS,
    FAST_CARB_K, FAST_CARB_THETA, PUBLIC_HOLIDAYS_PER_YEAR_MIN,
    PUBLIC_HOLIDAYS_PER_YEAR_MAX, SIMULATION_START_DAY_OF_WEEK,
    bolus_pk_for_dose, BOLUS_DIA_BASE_HOURS, BOLUS_DIA_MIN_HOURS, BOLUS_DIA_MAX_HOURS,
    GLYCOGEN_CAPACITY_GRAMS, DT_MINUTES,
    SEVERE_HYPO_REFRACTORY_MIN, HYPO_FOLLOWUP_SKILL_THRESHOLD,
    SEVERE_HYPO_THRESHOLD, ActiveCurve, STEPS_PER_DAY,
)


@pytest.fixture
def immortal(monkeypatch):
    """Disable death for tests about mechanics that need a full-length run."""
    monkeypatch.setattr(simulator, 'BG_DEATH_MGDL', -1e9)


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

    def test_insulin_lowers_bg(self, immortal, monkeypatch):
        """Injecting a large insulin curve into a high-BG simulation lowers BG."""
        monkeypatch.setattr(T1DMSimulator, '_check_and_correct', lambda self, idx: None)
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
    def test_day_of_week_cycles(self, immortal):
        """day_of_week increments correctly through the week."""
        sim = T1DMSimulator(seed=0)
        # Generate 7 days worth of steps; check day_of_week at each day boundary
        seen_days = set()
        for _ in range(7 * 288):
            step = sim.generate()
            seen_days.add(sim.state.day_of_week)
        assert seen_days == {0, 1, 2, 3, 4, 5, 6}, (
            f"Not all days of week seen: {seen_days}")

    def test_weekend_flag_matches_day_of_week(self, immortal):
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

    def test_glycogen_within_bounds_over_long_run(self, immortal):
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
        # Clear carbs-on-board so these tests exercise the refractory TIMER
        # alone. Otherwise the rule-of-15 recheck blocks the follow-up rescue
        # for its own (correct) reason and the timer is never reached.
        sim._rescue_totals[:] = 0.0

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
        # eff_low_thresh = 70 + 18*skill_avg ≥ 70, so 62 always triggers.
        assert first_idx == idx, (
            f"moderate-hypo correction failed to fire (last_hypo_correction_idx={first_idx})")

        # 15 min later — within 20-min moderate refractory, must block.
        # (15 not 11: int(11/5)=2 steps=10 min, which is *outside* the 10-min
        # severe gate too — the test would pass regardless of which refractory
        # is selected. 15 min is inside 20-min but outside 10-min.)
        idx2 = idx + int(15 / DT_MINUTES)
        sim.state.bg = sim.state.bg_observed = 62.0
        sim.state.last_cgm_check_idx = -9999
        sim._rescue_totals[:] = 0.0  # isolate the timer from the rule-of-15 recheck
        sim._check_and_correct(idx2)
        assert sim.state.last_hypo_correction_idx == first_idx, (
            "moderate hypo fired within 20 min — severe refractory used instead")

        # 25 min later — outside 20-min refractory, must release.
        idx3 = idx + int(25 / DT_MINUTES)
        sim.state.bg = sim.state.bg_observed = 62.0
        sim.state.last_cgm_check_idx = -9999
        sim._rescue_totals[:] = 0.0  # isolate the timer from the rule-of-15 recheck
        sim._check_and_correct(idx3)
        assert sim.state.last_hypo_correction_idx == idx3, (
            "moderate hypo did not re-fire after 25 min — refractory stuck")


class TestRuleOfFifteenRecheck:
    """Carbs-on-board gating of repeat rescues — the recheck half of rule-of-15.

    The refractory timer supplies the waiting; this supplies the "and only
    re-dose if still low". Without it a deep low cascades into a dose every
    10 minutes, blind to the glucose already absorbing.
    """

    def _low_patient(self, competence):
        sim = T1DMSimulator(seed=0)
        sim.generate()
        idx = sim.state.current_idx
        s = sim.state
        s.bg = s.bg_observed = 50.0
        s.last_cgm_check_idx = -9999
        s.last_hypo_correction_idx = -9999
        sim._today_wake_idx = 0
        sim._today_sleep_idx = idx + STEPS_PER_DAY
        sim.patient.dosing_competence = competence
        return sim, idx

    def test_pending_rescue_blocks_a_redundant_dose(self):
        """A competent patient with enough glucose already absorbing does not re-dose."""
        sim, idx = self._low_patient(competence=0.95)
        sim._rescue_totals[:] = 0.0
        # 20 g still to absorb: at BG_SCALE_FACTOR that is far more than the
        # ~30 mg/dL needed to clear the threshold from 50.
        sim._rescue_totals[idx:idx + 12] = 20.0 / 12.0
        sim._check_and_correct(idx)
        assert sim.state.last_hypo_correction_idx == -9999, (
            "re-dosed despite 20 g of rescue carbs already on board")

    def test_no_carbs_on_board_still_treats(self):
        """The gate must not suppress a genuine untreated low."""
        sim, idx = self._low_patient(competence=0.95)
        sim._rescue_totals[:] = 0.0
        sim._check_and_correct(idx)
        assert sim.state.last_hypo_correction_idx == idx, (
            "failed to treat a low with no carbs on board")

    def test_awareness_is_skill_scaled(self):
        """Low competence discounts carbs-on-board less, so it re-doses sooner."""
        cob, fired = 9.0, {}
        for comp in (0.15, 0.95):
            n = 0
            for seed in range(40):
                sim = T1DMSimulator(seed=seed)
                sim.generate()
                idx = sim.state.current_idx
                s = sim.state
                s.bg = s.bg_observed = 50.0
                s.last_cgm_check_idx = -9999
                s.last_hypo_correction_idx = -9999
                sim._today_wake_idx = 0
                sim._today_sleep_idx = idx + STEPS_PER_DAY
                sim.patient.dosing_competence = comp
                sim._rescue_totals[:] = 0.0
                sim._rescue_totals[idx:idx + 12] = cob / 12.0
                sim._check_and_correct(idx)
                n += sim.state.last_hypo_correction_idx == idx
            fired[comp] = n
        assert fired[0.15] > fired[0.95], (
            f"careless patients should re-dose more often on the same carbs-on-board "
            f"(low-skill {fired[0.15]}/40 vs high-skill {fired[0.95]}/40)")


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

    def test_followup_lifts_post_correction_bg_trace(self):
        """The followup snack must measurably raise BG over the post-correction
        window vs an otherwise-identical patient who skips the followup. This
        is the structural test for CLAUDE.md's warning that removing the tail
        re-opens 6+ hour dangerous hypos."""
        import simulator as _sim

        def run(seed: int, with_followup: bool) -> np.ndarray:
            # Toggle ONLY the follow-up snack (via its carb fraction), holding
            # the patient and skill fixed — comparing high- vs low-skill patients
            # confounds the snack with every other skill-dependent behaviour.
            _saved = _sim.HYPO_FOLLOWUP_FRACTION
            _sim.HYPO_FOLLOWUP_FRACTION = _saved if with_followup else 0.0
            try:
                sim = T1DMSimulator(seed=seed)
                p = sim.patient
                p.attentiveness = 0.9  # above HYPO_FOLLOWUP_SKILL_THRESHOLD so the snack is eaten
                p.dosing_competence = 0.9
                # Isolate the snack's effect from the always-on glucose-
                # effectiveness mean-reversion, which would clear the lift.
                p.glucose_effectiveness = 0.0
                return _run_body(sim)
            finally:
                _sim.HYPO_FOLLOWUP_FRACTION = _saved

        def _run_body(sim):
            sim._pending_events = []
            sim.state.active_curves = []
            sim.state.is_sick = False
            sim.generate()
            idx = sim.state.current_idx
            s = sim.state
            s.bg = 40.0
            s.bg_observed = 40.0
            sim._interstitial_bg = 40.0  # reset CGM interstitial-lag state to match the forced BG
            s.last_cgm_check_idx = -9999
            s.last_hypo_correction_idx = -9999
            sim._today_wake_idx = 0
            sim._today_sleep_idx = idx + STEPS_PER_DAY
            sim._check_and_correct(idx)
            bgs = []
            for _ in range(120 // DT_MINUTES):
                step = sim.generate()
                bgs.append(float(step['bg']))
            return np.array(bgs)

        # Average the effect over several seeds — the lift is a population
        # tendency, not a single-realization guarantee, so a per-seed threshold
        # is fragile to any RNG-stream change.
        seeds = [3, 5, 11, 17, 23, 29]
        lifts, min_gains = [], []
        for sd in seeds:
            bgs_with = run(sd, with_followup=True)
            bgs_without = run(sd, with_followup=False)
            lifts.append(bgs_with.mean() - bgs_without.mean())
            min_gains.append(bgs_with.min() - bgs_without.min())

        # With-followup trace should be higher on average over the 2h window:
        # the followup tail keeps glucose flowing while the rescue burst fades.
        assert np.mean(lifts) > 3.0, (
            f"followup did not lift the post-correction trace meaningfully: "
            f"mean lift over seeds = {np.mean(lifts):.1f} mg/dL (per-seed {[round(x,1) for x in lifts]})")
        # And the minimum dip should be less severe with followup, on average.
        assert np.mean(min_gains) > 0.0, (
            f"followup did not raise the post-correction minimum: "
            f"mean min-gain over seeds = {np.mean(min_gains):.1f}")


class TestSevereHypoRescueAmount:
    """CLAUDE.md structural rule: severe hypo triggers a non-probabilistic
    >=14g rescue that grows linearly with deficit (14 + 0.35 * deficit).
    The grams floor is what keeps severe episodes <= 1h."""

    def _force_correction_and_get_grams(self, sim, target_bg):
        sim.generate()
        idx = sim.state.current_idx
        s = sim.state
        s.bg = target_bg
        s.bg_observed = target_bg
        s.last_cgm_check_idx = -9999
        s.last_hypo_correction_idx = -9999
        sim._today_wake_idx = 0
        sim._today_sleep_idx = idx + STEPS_PER_DAY
        before = {id(c) for c in s.active_curves}
        sim._check_and_correct(idx)
        rescues = [c for c in s.active_curves
                   if c.curve_type == 'correction_carb'
                   and id(c) not in before
                   and 'followup' not in c.label.lower()]
        assert len(rescues) == 1, (
            f"expected exactly one rescue curve, got {len(rescues)}: "
            f"{[c.label for c in rescues]}")
        return float(np.sum(rescues[0].values))

    def test_rescue_minimum_is_at_least_14g(self):
        """Severe hypo with tiny deficit should still rescue with >=14g."""
        sim = T1DMSimulator(seed=0)
        # BG just below threshold: deficit ~= 0
        g = self._force_correction_and_get_grams(sim, SEVERE_HYPO_THRESHOLD - 0.5)
        assert g >= 14.0 - 1e-6, (
            f"rescue at BG={SEVERE_HYPO_THRESHOLD-0.5} returned {g:.2f}g, "
            f"below the 14g floor")

    def test_rescue_grams_grow_with_deficit(self):
        """Deeper hypo must yield strictly more carbs (deficit-driven term)."""
        # Use the same patient (same seed) so skill multiplier is constant.
        sim_shallow = T1DMSimulator(seed=4)
        sim_deep = T1DMSimulator(seed=4)
        g_shallow = self._force_correction_and_get_grams(sim_shallow, 50.0)
        g_deep = self._force_correction_and_get_grams(sim_deep, 30.0)
        # Difference attributable to deficit term:
        # delta_deficit = (54-30) - (54-50) = 20, so delta_g should be ~= 0.35 * 20 = 7g
        assert g_deep > g_shallow + 3.0, (
            f"BG=30 rescue ({g_deep:.2f}g) should clearly exceed BG=50 "
            f"rescue ({g_shallow:.2f}g) per the 14 + 0.35*deficit formula")


class TestHypoCorrectionSkillScaling:
    """CLAUDE.md structural rule: hypo correction grams scale with skill
    (skill_avg = (s2+s3)/2). Without this, high-skill patients
    under-correct and the population TBR ceiling sticks at ~30%."""

    def _moderate_hypo_grams(self, sim, attentiveness, dosing):
        p = sim.patient
        p.attentiveness = attentiveness
        p.dosing_competence = dosing
        # Cancel the rage-eat random branch by raising BG above
        # RAGE_EAT_BG_THRESHOLD but keeping it below eff_low_thresh.
        sim.generate()
        idx = sim.state.current_idx
        s = sim.state
        s.bg = 65.0
        s.bg_observed = 65.0
        s.last_cgm_check_idx = -9999
        s.last_hypo_correction_idx = -9999
        sim._today_wake_idx = 0
        sim._today_sleep_idx = idx + STEPS_PER_DAY
        before = {id(c) for c in s.active_curves}
        sim._check_and_correct(idx)
        rescues = [c for c in s.active_curves
                   if c.curve_type == 'correction_carb'
                   and id(c) not in before
                   and 'followup' not in c.label.lower()]
        assert len(rescues) == 1, (
            f"expected one rescue curve, got {len(rescues)}")
        return float(np.sum(rescues[0].values))

    def test_high_skill_corrects_with_more_grams(self):
        """High-skill (skill_avg=0.9) > low-skill (skill_avg=0.3) by the
        (1 + 1.5*skill_avg) multiplier — ratio ~= 2.35/1.45 ~= 1.6×."""
        sim_low = T1DMSimulator(seed=0)
        sim_high = T1DMSimulator(seed=0)  # same patient, only skills overridden
        g_low = self._moderate_hypo_grams(sim_low, 0.3, 0.3)
        g_high = self._moderate_hypo_grams(sim_high, 0.9, 0.9)
        assert g_high > g_low, (
            f"high-skill correction ({g_high:.2f}g) should exceed low-skill "
            f"({g_low:.2f}g) per the (1 + 1.5*skill_avg) multiplier")
        # Expected: low ~ HYPO_CORRECTION_BASE_GRAMS * 1.45,
        # high ~ HYPO_CORRECTION_BASE_GRAMS * 2.35 — ratio ~= 1.6.
        # Allow slack for the panic-factor severity term.
        assert g_high / g_low > 1.25, (
            f"ratio {g_high/g_low:.2f} too small — skill multiplier may be flat")


class TestBolusPKForDoseIntegration:
    """The dose-dependent bolus PK helper is unit-tested above; this verifies
    the simulator's bolus dispatch *actually routes through it*, rather than
    the legacy fixed-duration BOLUS_DURATION_HOURS path."""

    def test_meal_boluses_use_bolus_pk_for_dose(self, monkeypatch):
        """At least a few meal/correction boluses fire through the helper
        within a 48h run, and the helper's duration response monotonically
        scales with dose (smallest observed dose ≤ largest observed duration)."""
        import simulator as sim_module
        real_helper = sim_module.bolus_pk_for_dose
        calls = []

        def spy(dose, *args, **kwargs):
            result = real_helper(dose, *args, **kwargs)
            calls.append((float(dose), result))
            return result

        monkeypatch.setattr(sim_module, 'bolus_pk_for_dose', spy)

        sim = sim_module.T1DMSimulator(seed=0)
        sim.generate_hours(48)

        assert len(calls) >= 3, (
            f"only {len(calls)} bolus_pk_for_dose calls in 48h — "
            "simulator may be bypassing the dose-dependent PK helper")

        # If the run produced clearly different doses, the helper must have
        # responded with non-decreasing duration. (The unit-level monotonicity
        # is already pinned by TestBolusPKForDose; this catches the case where
        # the simulator stops calling the helper and falls back to a constant.)
        by_dose = sorted(calls)
        if by_dose[-1][0] / max(by_dose[0][0], 1e-9) > 2.0:
            small_dur = by_dose[0][1][2]
            large_dur = by_dose[-1][1][2]
            assert large_dur >= small_dur - 1e-9, (
                f"larger bolus did not yield equal-or-longer duration: "
                f"dose={by_dose[0][0]:.2f} dur={small_dur:.1f} vs "
                f"dose={by_dose[-1][0]:.2f} dur={large_dur:.1f}")


class TestBasalInjectionCadence:
    """Basal injection cadence must equal patient.basal_dose_interval_hours.

    Real glargine and degludec are dosed once daily, so the cadence is 24h
    regardless of the analogue's PK action duration (glargine 26h, degludec
    42h). Each once-daily dose delivers the full 24h basal_dose, keeping the
    24h-average insulin delivery aligned with `basal_dose`.
    """

    def test_basal_cadence_matches_patient_duration(self, immortal):
        for seed in [0, 5, 13, 27]:
            sim = T1DMSimulator(seed=seed)
            basal_indices: list = []
            original_inject = sim.inject_curve

            def capturing_inject(values, start_idx, curve_type, label='',
                                  _orig=original_inject, _store=basal_indices):
                if curve_type == 'basal':
                    _store.append(int(start_idx))
                return _orig(values, start_idx, curve_type, label)

            sim.inject_curve = capturing_inject

            # 14 days covers ~14 injections for an 18h patient, ~11 for a 30h
            # patient — plenty of samples for a robust median.
            sim.generate_hours(14 * 24)

            assert len(basal_indices) >= 8, (
                f"seed={seed}: only {len(basal_indices)} basals scheduled in "
                f"14 days (patient duration={sim.patient.basal_duration_hours:.2f}h)")

            spacings_steps = np.diff(sorted(basal_indices))
            median_spacing_h = float(np.median(spacings_steps)) * DT_MINUTES / 60.0
            # Per-dose ±30 min jitter + max(current_idx, ...) clamp can drift
            # individual spacings; the median should still land within 1.5h.
            assert abs(median_spacing_h - sim.patient.basal_dose_interval_hours) < 1.5, (
                f"seed={seed}: median basal spacing {median_spacing_h:.2f}h "
                f"!= patient.basal_dose_interval_hours "
                f"{sim.patient.basal_dose_interval_hours:.2f}h")


class TestInjectCurveUpdatesTotals:
    """CLAUDE.md: 'Use inject_curve() (not state.active_curves.append) whenever
    inserting curves from outside generate().' Raw append leaves the
    pre-accumulated _carb/_basal/_bolus/_exercise totals stale, so per-step
    reads silently drop the contribution. This test pins that contract."""

    def test_inject_curve_changes_bg_response(self):
        """A carb curve injected via inject_curve must visibly raise BG;
        the same curve appended raw to active_curves must NOT (because the
        totals arrays — which drive the per-step reads — stay untouched)."""
        # Path A: use inject_curve
        sim_a = T1DMSimulator(seed=11, initial_bg=120.0)
        for _ in range(10):
            sim_a.generate()
        bg_before_a = sim_a.state.bg
        curve = gamma_curve(60.0, FAST_CARB_K, FAST_CARB_THETA, 120.0)
        sim_a.inject_curve(curve, sim_a.state.current_idx, 'carb', 'injected')
        for _ in range(12):
            sim_a.generate()
        delta_a = sim_a.state.bg - bg_before_a

        # Path B: raw append (the contract violation)
        sim_b = T1DMSimulator(seed=11, initial_bg=120.0)
        for _ in range(10):
            sim_b.generate()
        bg_before_b = sim_b.state.bg
        sim_b.state.active_curves.append(ActiveCurve(
            start_time_idx=sim_b.state.current_idx,
            values=curve.copy(),
            curve_type='carb',
            label='raw-append',
        ))
        for _ in range(12):
            sim_b.generate()
        delta_b = sim_b.state.bg - bg_before_b

        # Path A should produce a real BG rise; path B should not (because the
        # carb contribution is read from _carb_totals, which raw append doesn't
        # update). This is the bug-class the CLAUDE.md warning prevents.
        assert delta_a > delta_b + 10.0, (
            f"inject_curve effect (Δ={delta_a:+.1f}) should clearly exceed "
            f"raw-append effect (Δ={delta_b:+.1f}). Either the totals arrays "
            f"are being bypassed, or raw append is being silently honored — "
            f"both violate the documented contract.")
