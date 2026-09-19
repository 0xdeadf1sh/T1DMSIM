"""
Tests for the carbohydrate the simulated patient LOGS, as distinct from the truth.
Every carb event is logged at its true onset step; its grams and GI are guesses.
The logged curve is exported as `total_carb`; only `total_carb_true` drives BG.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import simulator as S  # noqa: E402
from simulator import (CARB_COUNT_UNDERBOLUS_BIAS, GI_GUESS_SIGMA_BASE,
                       STEPS_PER_DAY, T1DMSimulator, gamma_curve,
                       gi_gamma_params)  # noqa: E402


def _run(seed: int, steps: int = 4 * STEPS_PER_DAY, cls=T1DMSimulator) -> dict:
    sim = cls(seed=seed)
    out = [sim.generate() for _ in range(steps)]
    return {k: np.array([s[k] for s in out]) for k in
            ('bg', 'bg_observed', 'total_insulin', 'total_carb',
             'total_carb_true', 'carb_g', 'carb_gi')}


def test_gram_guess_ratio_matches_the_patients_carb_count_sigma():
    """A non-meal carb event's logged grams are true grams x (1 + N(bias, sigma))."""
    sim = T1DMSimulator(seed=5)
    sim.patient.carb_count_error_sigma = 0.25
    sim.patient.gi_guess_sigma = 10.0
    draws = [sim._logged_carb(20.0, 50.0)[1] for _ in range(40000)]
    ratios = np.array([(e['dose'] if e else 0.0) / 20.0 for e in draws])
    assert ratios.mean() == pytest.approx(1.0 + CARB_COUNT_UNDERBOLUS_BIAS, abs=0.01)
    assert ratios.std(ddof=1) == pytest.approx(0.25, abs=0.01)


def test_gi_guess_sigma_is_the_carb_count_shape_on_s3():
    for seed in range(12):
        p = T1DMSimulator(seed=seed).patient
        s3 = p.dosing_competence
        assert p.gi_guess_sigma == pytest.approx(GI_GUESS_SIGMA_BASE * (1.3 - s3) ** 2)
        assert p.gi_guess_sigma > 0.0


def test_gi_guess_spread_matches_the_sigma_and_clips_to_the_scale():
    sim = T1DMSimulator(seed=6)
    sim.patient.carb_count_error_sigma = 0.0
    sim.patient.gi_guess_sigma = 10.0
    mid = np.array([sim._logged_carb(30.0, 50.0)[1]['gi'] for _ in range(40000)])
    assert mid.mean() == pytest.approx(50.0, abs=0.2)
    assert mid.std(ddof=1) == pytest.approx(10.0, abs=0.2)

    sim.patient.gi_guess_sigma = 40.0
    hi = np.array([sim._logged_carb(30.0, 100.0)[1]['gi'] for _ in range(4000)])
    lo = np.array([sim._logged_carb(30.0, 0.0)[1]['gi'] for _ in range(4000)])
    assert hi.max() == 100.0 and hi.min() >= 0.0
    assert lo.min() == 0.0 and lo.max() <= 100.0
    assert (hi == 100.0).mean() > 0.3 and (lo == 0.0).mean() > 0.3


def test_a_zero_gram_guess_logs_nothing():
    sim = T1DMSimulator(seed=1)
    assert sim._logged_carb(0.0, 50.0) == (None, None)
    curve, event = sim._logged_carb(30.0, 50.0, logged_grams=0.0)
    assert curve is None and event is None
    idx = sim.state.current_idx
    sim.inject_curve(gamma_curve(30.0, *gi_gamma_params(50.0)), idx, 'carb',
                     event=event, logged=curve)
    step = sim.generate()
    assert step['total_carb'] == 0.0 and step['carb_g'] == 0.0
    assert step['total_carb_true'] > 0.0


def test_logged_curve_sums_to_the_guess_at_the_true_onset():
    sim = T1DMSimulator(seed=2)
    idx = sim.state.current_idx + 10
    curve, event = sim._logged_carb(40.0, 60.0)
    sim.inject_curve(gamma_curve(40.0, *gi_gamma_params(60.0)), idx, 'carb',
                     event=event, logged=curve)
    logged = sim._logged_carb_totals
    assert logged[:idx].sum() == 0.0
    assert logged[idx:].sum() == pytest.approx(event['dose'])
    assert sim._carb_totals[idx:].sum() == pytest.approx(40.0)


class _NoLog(T1DMSimulator):
    """A patient who logs nothing at all."""

    def _logged_carb(self, grams, gi, logged_grams=None):
        return None, None


def test_bg_is_identical_when_nothing_is_logged():
    for seed in (0, 3, 11):
        base = _run(seed)
        none = _run(seed, cls=_NoLog)
        assert np.array_equal(base['bg'], none['bg'])
        assert np.array_equal(base['bg_observed'], none['bg_observed'])
        assert np.array_equal(base['total_insulin'], none['total_insulin'])
        assert np.array_equal(base['total_carb_true'], none['total_carb_true'])
        assert none['total_carb'].sum() == 0.0 and none['carb_g'].sum() == 0.0
        assert base['total_carb_true'].sum() > 0.0


def test_bg_is_identical_when_the_gi_guess_widens():
    saved = S.GI_GUESS_SIGMA_BASE
    try:
        base = _run(3)
        S.GI_GUESS_SIGMA_BASE = 45.0
        wide = _run(3)
    finally:
        S.GI_GUESS_SIGMA_BASE = saved
    assert np.array_equal(base['bg'], wide['bg'])
    assert np.array_equal(base['total_carb_true'], wide['total_carb_true'])
    assert not np.array_equal(base['carb_gi'], wide['carb_gi'])


def test_bg_is_identical_when_the_log_stream_changes():
    saved = S.CARB_LOG_SEED_STREAM
    try:
        base = _run(3)
        S.CARB_LOG_SEED_STREAM = saved + 1
        other = _run(3)
    finally:
        S.CARB_LOG_SEED_STREAM = saved
    assert np.array_equal(base['bg'], other['bg'])
    assert np.array_equal(base['total_carb_true'], other['total_carb_true'])
    assert not np.array_equal(base['total_carb'], other['total_carb'])
    assert base['total_carb'].sum() > 0.0


def test_logged_carbs_differ_from_the_truth_but_track_it():
    for seed in (0, 3, 11):
        d = _run(seed, steps=20 * STEPS_PER_DAY)
        true, logged = d['total_carb_true'].sum(), d['total_carb'].sum()
        assert logged > 0.0 and not np.array_equal(d['total_carb'], d['total_carb_true'])
        assert logged == pytest.approx(true, rel=0.35)
        assert d['carb_g'].sum() == pytest.approx(logged, rel=0.1)


def test_meal_log_equals_the_carb_count_behind_its_bolus():
    for seed in (0, 4, 9, 17):
        sim = T1DMSimulator(seed=seed)
        sim._pending_events = []
        sim._site_quality = lambda s4: 1.0  # isolate the dose from site-absorption noise
        sim._generate_day_events()
        icr = sim.patient.icr
        meals = [d['event']['dose'] for _, t, d in sim._pending_events
                 if t == 'carb' and d.get('event') is not None]
        boluses = [d['event']['dose'] * icr for _, t, d in sim._pending_events
                   if t == 'bolus']
        assert meals and boluses
        for b in boluses:
            assert any(abs(b - m) <= 1e-9 * max(1.0, m) for m in meals), (
                f'seed={seed}: bolus for {b:.6f} g has no matching logged meal {meals}')


def test_cache_channel_carries_the_logged_carbs():
    pytest.importorskip('blosc2')
    import cache_simulator as cs
    cfg = cs.RowConfig(
        warmup_steps=0, keep_steps=3 * STEPS_PER_DAY, max_attempts=1,
        rail_high=1e9, rail_low=-1e9, hypo_prob=0.0, hypo_min_frac=0.0,
        hypo_threshold=70.0, seed_salt=0, event_refractory_steps=6, events=True)
    channels, _ = cs._simulate_patient(7, cfg)
    d = _run(7, steps=3 * STEPS_PER_DAY)
    assert np.array_equal(channels['total_carb'], d['total_carb'].astype(np.float32))
    assert np.array_equal(channels['carb_g'], d['carb_g'].astype(np.float32))
    assert not np.array_equal(channels['total_carb'],
                              d['total_carb_true'].astype(np.float32))
    assert 'total_carb_true' not in channels
