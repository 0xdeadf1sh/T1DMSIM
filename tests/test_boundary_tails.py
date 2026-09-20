"""Tests for the behaviour-off boundary and the four counterfactual tails.

The geometry is SPEC/cache.md. These pin the lockstep between the arms, what the
log carries versus what drives BG, that nothing is chopped at the boundary, and
that a tail is never rejected."""

import copy
import json
import os
import sys

import numpy as np
import pytest

pytest.importorskip("blosc2")  # cache_simulator imports blosc2
import blosc2  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cache_simulator as cs  # noqa: E402
from simulator import (BG_CLAMP_MIN, BOLUS_VARIANTS, STEPS_PER_DAY,  # noqa: E402
                       T1DMSimulator, bolus_pk_for_dose, gamma_curve,
                       gi_gamma_params)


_CFG = dict(
    warmup_steps=STEPS_PER_DAY, context_steps=288, max_attempts=1,
    rail_high=1e9, rail_low=-1e9, hypo_prob=0.0, hypo_min_frac=0.0,
    hypo_threshold=70.0, seed_salt=0, event_refractory_steps=6, events=True,
)


def _cfg(**kw) -> cs.RowConfig:
    return cs.RowConfig(**(_CFG | kw))


def _boundary(seed: int, cfg: cs.RowConfig):
    """(simulator at the boundary, context, dose) with the row's draws applied."""
    sim, ctx, _, draws = cs.simulate_to_boundary(seed, cfg)
    return sim, ctx, cs.boundary_dose(sim, draws)


def test_deepcopy_reproduces_the_simulator():
    sim = T1DMSimulator(seed=3)
    for _ in range(600):
        sim.generate()
    a, b = copy.deepcopy(sim), copy.deepcopy(sim)
    xa = [a.generate()['bg'] for _ in range(80)]
    xb = [b.generate()['bg'] for _ in range(80)]
    assert xa == xb
    assert a.state.current_idx == b.state.current_idx
    assert xa == [sim.generate()['bg'] for _ in range(80)], 'the parent is untouched'


def test_behaviour_off_generates_no_events_over_48h():
    sim = T1DMSimulator(seed=5)
    for _ in range(600):
        sim.generate()
    off = copy.deepcopy(sim)
    off.behaviour_enabled = False
    n_pending = len(off._pending_events)
    n_curves = len(off.state.active_curves)
    for _ in range(2 * STEPS_PER_DAY):
        s = off.generate()
        assert all(s[c] == 0.0 for c in cs.EVENT_CHANNELS)
    assert len(off._pending_events) == n_pending, 'no pending event was activated'
    assert len(off.state.active_curves) <= n_curves, 'no curve was injected'


def test_behaviour_off_carries_the_wake_window_across_midnight():
    sim = T1DMSimulator(seed=9)
    for _ in range(600):
        sim.generate()
    off = copy.deepcopy(sim)
    off.behaviour_enabled = False
    wake, sleep = off._today_wake_idx, off._today_sleep_idx
    day = off.state.day_number
    for _ in range(2 * STEPS_PER_DAY):
        off.generate()
    rolled = off.state.day_number - day
    assert rolled >= 1, 'the day must still roll over with behaviour off'
    assert off._today_wake_idx == wake + rolled * STEPS_PER_DAY
    assert off._today_sleep_idx == sleep + rolled * STEPS_PER_DAY
    assert off._today_wake_idx < off.state.current_idx + STEPS_PER_DAY


def test_arm0_matches_an_untouched_behaviour_off_continuation():
    cfg = _cfg()
    sim, _, _, draws = cs.simulate_to_boundary(101, cfg)
    ref = copy.deepcopy(sim)          # taken before the boundary draws, not after
    ref.behaviour_enabled = False
    ref_bg = np.array([ref.generate()['bg'] for _ in range(cfg.tail_steps)], np.float32)
    tails = cs.run_tails(sim, cs.boundary_dose(sim, draws), cfg)
    assert np.array_equal(tails['bg'][0], ref_bg), 'arm 0 must be bit-identical'


def test_boundary_dose_leaves_the_parent_streams_where_the_context_ended():
    cfg = _cfg()
    sim, _, _, draws = cs.simulate_to_boundary(106, cfg)
    before = (sim.rng.bit_generator.state, sim._log_rng.bit_generator.state)
    dose = cs.boundary_dose(sim, draws)
    assert (sim.rng.bit_generator.state, sim._log_rng.bit_generator.state) == before
    assert dose.site_factor != 1.0
    assert dose.logged_carb_g != dose.carb_g
    assert cs.boundary_dose(sim, draws) == dose, 'so the draws stay repeatable'


def test_arms_differ_only_through_the_injected_dose():
    cfg = _cfg()
    sim, _, dose = _boundary(102, cfg)
    tails = cs.run_tails(sim, dose, cfg)
    T = cfg.tail_steps

    av = BOLUS_VARIANTS[dose.analogue]
    k, theta, dur = bolus_pk_for_dose(dose.bolus_u, av['gamma_k'], av['gamma_theta'],
                                      av['dia_base_hours'])
    delivered = gamma_curve(dose.bolus_u * dose.site_factor, k, theta, dur)
    logged_bolus = delivered / dose.site_factor
    logged_carb = gamma_curve(dose.logged_carb_g, *gi_gamma_params(dose.logged_carb_gi))

    ins, carb = tails['bolus_insulin'], tails['total_carb']
    assert np.allclose(ins[1] - ins[0], logged_bolus[:T], rtol=1e-4, atol=1e-6)
    assert np.allclose(carb[2] - carb[0], logged_carb[:T], rtol=1e-4, atol=1e-6)
    # An arm without a dose of a kind carries exactly arm 0's channel for it.
    assert np.array_equal(carb[1], carb[0])
    assert np.array_equal(ins[2], ins[0])
    for arm in (1, 2, 3):
        assert not np.array_equal(tails['bg'][arm], tails['bg'][0])


def test_arm3_reuses_arm1_bolus_and_arm2_carbs():
    cfg = _cfg()
    sim, _, dose = _boundary(103, cfg)
    tails = cs.run_tails(sim, dose, cfg)
    ins, carb = tails['bolus_insulin'], tails['total_carb']
    assert np.allclose(ins[3] - ins[0], ins[1] - ins[0], rtol=1e-6, atol=1e-7)
    assert np.allclose(carb[3] - carb[0], carb[2] - carb[0], rtol=1e-6, atol=1e-7)
    assert tails['bolus_u'][3, 0] == tails['bolus_u'][1, 0]
    assert tails['carb_g'][3, 0] == tails['carb_g'][2, 0]
    assert tails['carb_gi'][3, 0] == tails['carb_gi'][2, 0]


def test_bolus_event_carries_the_intended_dose_while_delivered_drives_bg():
    cfg = _cfg()
    sim, _, dose = _boundary(104, cfg)
    c = copy.deepcopy(sim)
    c.behaviour_enabled = False
    i = c.state.current_idx
    span = slice(i, i + 200)
    true_before = float(c._bolus_totals[span].sum())
    log_before = float(c._logged_bolus_totals[span].sum())

    c.inject_bolus(dose.bolus_u, dose.analogue, dose.site_factor, start_idx=i)
    delivered = dose.bolus_u * dose.site_factor
    assert float(c._bolus_totals[span].sum()) - true_before == pytest.approx(delivered, rel=1e-9)
    assert float(c._logged_bolus_totals[span].sum()) - log_before == pytest.approx(
        dose.bolus_u, rel=1e-9)
    assert dose.site_factor != 1.0
    assert c.generate()['bolus_u'] == pytest.approx(dose.bolus_u, rel=1e-6)


def test_tail_logged_carbs_differ_from_the_true_carbs():
    cfg = _cfg()
    sim, _, dose = _boundary(105, cfg)
    assert dose.logged_carb_g != dose.carb_g
    c = copy.deepcopy(sim)
    c.behaviour_enabled = False
    i = c.state.current_idx
    span = slice(i, i + 200)
    true_before = float(c._carb_totals[span].sum())
    log_before = float(c._logged_carb_totals[span].sum())
    c.inject_carbs(dose.carb_g, dose.carb_gi, dose.logged_carb_g, dose.logged_carb_gi,
                   start_idx=i)
    assert float(c._carb_totals[span].sum()) - true_before == pytest.approx(
        dose.carb_g, rel=1e-9)
    assert float(c._logged_carb_totals[span].sum()) - log_before == pytest.approx(
        dose.logged_carb_g, rel=1e-9)


def test_tail_step_zero_carries_the_context_forward():
    """Doses from before the boundary keep absorbing; nothing is chopped."""
    cfg = _cfg(context_steps=576)
    found = False
    for seed in range(12):
        sim, ctx, dose = _boundary(cs._row_seed(seed, 0, 0), cfg)
        idx = sim.state.current_idx
        carried_ins = float(sim._logged_basal_totals[idx] + sim._logged_bolus_totals[idx])
        carried_carb = float(sim._logged_carb_totals[idx])
        tails = cs.run_tails(sim, dose, cfg)
        # Arm 0 reads the accumulators built before the boundary, untruncated.
        assert tails['total_insulin'][0, 0] == pytest.approx(carried_ins, rel=1e-6)
        assert tails['total_carb'][0, 0] == pytest.approx(carried_carb, rel=1e-6)
        assert carried_ins > 0.0, 'basal is always on board'
        assert carried_ins < 3.0 * float(ctx['total_insulin'][-1])
        if ctx['total_carb'][-1] > 0.05:
            found = True
            assert carried_carb > 0.0
            assert carried_carb < 3.0 * float(ctx['total_carb'][-1])
    assert found, 'no seed had carbohydrate on board at the boundary'


def test_no_tail_is_ever_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, 'TAIL_BOLUS_U_RANGE', (20.0, 20.0))
    out = str(tmp_path / 'railed')
    report = cs.build_cache(
        out_dir=out, pool_size=6, context_steps=576, warmup_hours=24.0, n_jobs=1,
        dataset_md='', baseline_stats=None, seed_salt=0, events=True,
    )
    assert report['corpus']['n_patients'] == 6
    assert report['sampling']['n_fallback_rows'] == 0

    tail_bg = np.asarray(blosc2.open(os.path.join(out, 'tail_bg_observed.b2nd'), mode='r'))
    ctx_bg = np.asarray(blosc2.open(os.path.join(out, 'bg_observed.b2nd'), mode='r'))
    rail_low = report['params']['rail_low']
    assert ctx_bg.min() > rail_low, 'every stored context clears the rail'
    assert tail_bg.min() <= rail_low, 'a 20 U tail crosses the rail and is kept'
    assert tail_bg.min() == pytest.approx(BG_CLAMP_MIN, abs=1e-3)


def test_tail_dose_arrays_are_the_boundary_step(tmp_path):
    out = str(tmp_path / 'doses')
    cs.build_cache(out_dir=out, pool_size=4, context_steps=288, warmup_hours=5.0,
                   n_jobs=1, dataset_md='', baseline_stats=None, events=True)
    for name in cs.EVENT_CHANNELS:
        step0 = np.asarray(blosc2.open(os.path.join(out, f'tail_{name}.b2nd'), mode='r'))[:, :, 0]
        dose = np.asarray(blosc2.open(os.path.join(out, f'tail_dose_{name}.b2nd'), mode='r'))
        assert dose.shape == (4, cs.N_TAIL_ARMS)
        assert np.array_equal(dose, step0)
    bolus = np.asarray(blosc2.open(os.path.join(out, 'tail_dose_bolus_u.b2nd'), mode='r'))
    carb = np.asarray(blosc2.open(os.path.join(out, 'tail_dose_carb_g.b2nd'), mode='r'))
    assert np.all(bolus[:, 0] == 0.0) and np.all(carb[:, 0] == 0.0)
    assert np.all(bolus[:, 1] > 0.0) and np.all(carb[:, 1] == 0.0)
    assert np.all(bolus[:, 2] == 0.0) and np.all(carb[:, 2] > 0.0)
    assert np.all(bolus[:, 3] > 0.0) and np.all(carb[:, 3] > 0.0)


def test_array_shapes_on_disk(tmp_path):
    out = str(tmp_path / 'shapes')
    cs.build_cache(out_dir=out, pool_size=4, context_steps=288, warmup_hours=5.0,
                   n_jobs=1, dataset_md='', baseline_stats=None, events=True)
    for name in cs.channel_names(True):
        arr = np.asarray(blosc2.open(os.path.join(out, f'{name}.b2nd'), mode='r'))
        assert arr.shape == (4, 288)
    for name in cs.tail_channel_names(True):
        arr = np.asarray(blosc2.open(os.path.join(out, f'tail_{name}.b2nd'), mode='r'))
        assert arr.shape == (4, cs.N_TAIL_ARMS, cs.DEFAULT_TAIL_STEPS)


def test_skills_round_trip(tmp_path):
    out = str(tmp_path / 'skills')
    cs.build_cache(out_dir=out, pool_size=4, context_steps=288, warmup_hours=5.0,
                   n_jobs=1, dataset_md='', baseline_stats=None, seed_salt=3,
                   rail_low=-1e9, rail_high=1e9)
    skills = np.load(os.path.join(out, cs.SKILLS_FILE))
    assert skills.shape == (4, len(cs.SKILL_NAMES))
    assert skills.dtype == np.float32
    for row in range(4):
        p = T1DMSimulator(seed=cs._row_seed(row, 0, 3)).patient
        want = np.array([getattr(p, n) for n in cs.SKILL_NAMES], np.float32)
        assert np.array_equal(skills[row], want)
    with open(os.path.join(out, 'meta.json')) as f:
        assert json.load(f)['params']['skills'] == list(cs.SKILL_NAMES)


def test_start_hour_varies_across_rows(tmp_path):
    offsets = {cs.row_draws(cs._row_seed(i, 0, 0), _cfg()).warmup_offset for i in range(24)}
    assert len(offsets) >= 10, 'the boundary hour must not be fixed'
    out = str(tmp_path / 'hours')
    cs.build_cache(out_dir=out, pool_size=8, context_steps=288, warmup_hours=5.0,
                   n_jobs=1, dataset_md='', baseline_stats=None)
    hod = np.asarray(blosc2.open(os.path.join(out, 'hour_of_day.b2nd'), mode='r'))[:, 0]
    assert len(set(hod.tolist())) >= 3


def test_row_rebuild_is_deterministic():
    cfg = _cfg()
    a, sa = cs.simulate_row(4242, cfg)
    b, sb = cs.simulate_row(4242, cfg)
    assert set(a) == set(b)
    for name in a:
        assert np.array_equal(a[name], b[name]), name
    for key in ('dose_bolus_u', 'dose_carb_g', 'dose_carb_gi', 'dose_site_factor'):
        assert sa[key] == sb[key]


def test_dose_stream_is_independent_of_the_simulator():
    # row_draws touches no simulator, so the tails cannot shift the population.
    cfg = _cfg()
    first = cs.row_draws(77, cfg)
    T1DMSimulator(seed=77).generate()
    assert cs.row_draws(77, cfg) == first
    assert cs.TAIL_BOLUS_U_RANGE[0] <= first.bolus_u <= cs.TAIL_BOLUS_U_RANGE[1]
    assert cs.TAIL_CARB_G_RANGE[0] <= first.carb_g <= cs.TAIL_CARB_G_RANGE[1]
    assert 0.0 <= first.carb_gi <= 100.0
    assert first.analogue in BOLUS_VARIANTS


def test_dataset_md_reports_the_tails(tmp_path):
    report = cs.build_cache(
        out_dir=str(tmp_path / 'md'), pool_size=4, context_steps=288, warmup_hours=5.0,
        n_jobs=1, dataset_md='', baseline_stats=None,
    )
    md = cs._render_dataset_md(report)
    assert '## Counterfactual tails' in md
    for arm in cs.TAIL_ARMS:
        assert f'| `{arm}` |' in md
    for m in cs.TAIL_REPORT_OFFSETS_MIN:
        assert f'+{m} min' in md
    assert 'Bolus (U, intended)' in md
    # Carbohydrate raises BG at +120 min and a bolus lowers it.
    bg = report['tails']['bg_mean']
    assert bg[2][-1] > bg[0][-1] > bg[1][-1]
