"""
Tests for the DATASET.md distribution comparison and hypo-oversampling shift.

cache_simulator.py folds a "Distribution vs the baseline simulator" comparison
into DATASET.md: it pools the cache's bg_observed during the transcode pass and
compares its moments / percentiles / band fractions / LBGI-HBGI against the
unbiased-simulator baseline stored in diff/stats.json (datasets.Sim). These
tests verify the streaming stats are exact, the baseline loads, and that
oversampling shifts the pooled distribution toward hypoglycemia in the expected
direction (and that an unbiased build sits on the baseline).
"""

import os
import sys

import numpy as np
import pytest

pytest.importorskip("blosc2")  # cache_simulator imports blosc2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cache_simulator as cs  # noqa: E402


# ---------------------------------------------------------------------------
# Pure-function unit tests (fast, no simulation)
# ---------------------------------------------------------------------------

def test_kovatchev_numerators_match_reference():
    rng = np.random.default_rng(0)
    bg = rng.uniform(40, 400, size=5000)
    sum_rl, sum_rh = cs._kovatchev_numerators(bg)
    f = 1.509 * (np.log(np.clip(bg, 1.0, None)) ** 1.084 - 5.381)
    assert sum_rl == pytest.approx((10 * np.minimum(f, 0) ** 2).sum())
    assert sum_rh == pytest.approx((10 * np.maximum(f, 0) ** 2).sum())


def test_hist_percentile_matches_numpy():
    rng = np.random.default_rng(1)
    bg = rng.uniform(45, 395, size=200000)
    hist = np.histogram(bg, bins=cs.BG_HIST_EDGES)[0].astype(float)
    for q in (5, 25, 50, 75, 95):
        got = cs._hist_percentile(hist, cs.BG_HIST_EDGES, q)
        assert got == pytest.approx(np.percentile(bg, q), abs=1.0)  # 1 mg/dL bins


def test_pool_distribution_matches_numpy():
    rng = np.random.default_rng(2)
    bg = rng.normal(160, 55, size=100000).clip(40, 400)
    hist = np.histogram(bg, bins=cs.BG_HIST_EDGES)[0].astype(float)
    rl, rh = cs._kovatchev_numerators(bg)
    bg_dist = {
        'sum': float(bg.sum()), 'sum_sq': float((bg * bg).sum()),
        'sum_cube': float((bg ** 3).sum()), 'sum_rl': rl, 'sum_rh': rh,
        'n': bg.size, 'hist': hist.tolist(),
    }
    d = cs._pool_distribution(bg_dist)
    assert d['mean'] == pytest.approx(bg.mean())
    assert d['std'] == pytest.approx(bg.std())
    assert d['skew'] == pytest.approx(float(((bg - bg.mean()) ** 3).mean() / bg.std() ** 3), abs=1e-6)
    assert d['median'] == pytest.approx(np.median(bg), abs=1.0)
    assert d['lbgi'] == pytest.approx((10 * np.minimum(1.509 * (np.log(bg) ** 1.084 - 5.381), 0) ** 2).mean())


def test_load_baseline_stats():
    base = cs._load_baseline_stats(cs.DEFAULT_BASELINE_STATS)
    assert base is not None, "diff/stats.json should be present and parseable"
    for k in ('mean', 'std', 'median', 'p5', 'p95', 'lbgi', 'hbgi',
              'frac_below54', 'frac_below70', 'frac_in_range', 'frac_above180'):
        assert k in base
    assert 120 < base['mean'] < 200          # plausible sim mean BG
    assert 0 < base['frac_below70'] < 0.2
    # The three coarse bands partition the axis (to within rounding of the split).
    assert base['frac_below70'] + base['frac_in_range'] + base['frac_above180'] == pytest.approx(1.0, abs=0.02)


def test_load_baseline_missing_returns_none():
    assert cs._load_baseline_stats("/no/such/stats.json") is None


# ---------------------------------------------------------------------------
# End-to-end build tests (small pools; use the real diff/stats.json baseline)
# ---------------------------------------------------------------------------

def _build(tmp_path, name, **kw):
    defaults = dict(
        pool_size=36, sim_hours=18.0, warmup_hours=6.0, n_jobs=1,
        dataset_md='', seed_salt=7,
    )
    defaults.update(kw)
    return cs.build_cache(out_dir=str(tmp_path / name), **defaults)


@pytest.fixture(scope="module")
def reports(tmp_path_factory):
    tp = tmp_path_factory.mktemp("caches")
    base = _build(tp, "unbiased", hypo_oversample=0.0)
    over = _build(tp, "oversampled", hypo_oversample=1.0, hypo_min_frac=0.06)
    return base, over


def test_report_carries_distribution_and_baseline(reports):
    base, _ = reports
    assert base['baseline'] is not None
    for k in ('mean', 'std', 'median', 'p5', 'p95', 'lbgi', 'hbgi'):
        assert k in base['distribution']
        assert k in base['baseline']


def test_oversampling_shifts_toward_hypo_vs_baseline(reports):
    _, over = reports
    b = over['baseline']
    assert over['glycemia']['frac_below70'] > b['frac_below70']
    assert over['distribution']['lbgi'] > b['lbgi']          # hypo risk rises
    assert over['distribution']['mean'] < b['mean']          # mean BG falls


def test_oversampled_below_baseline_for_hyper(reports):
    _, over = reports
    b = over['baseline']
    assert over['glycemia']['frac_above180'] < b['frac_above180']
    assert over['distribution']['hbgi'] < b['hbgi']


def test_oversampled_shifts_relative_to_unbiased(reports):
    base, over = reports
    assert over['glycemia']['frac_below70'] > base['glycemia']['frac_below70']
    assert over['distribution']['lbgi'] > base['distribution']['lbgi']
    assert over['distribution']['mean'] < base['distribution']['mean']


def test_unbiased_build_sits_near_baseline(reports):
    base, _ = reports
    b = base['baseline']
    # A small unbiased pool over a short window won't match a 70-day baseline
    # exactly, but should be in the same neighbourhood.
    assert abs(base['distribution']['mean'] - b['mean']) < 12.0
    assert abs(base['glycemia']['frac_below70'] - b['frac_below70']) < 0.05


def test_no_baseline_flag_omits_comparison(tmp_path):
    rep = _build(tmp_path, "nb", hypo_oversample=0.5, baseline_stats=None)
    assert rep['baseline'] is None
    assert "Distribution vs the baseline simulator" not in cs._render_dataset_md(rep)


def test_comparison_section_rendered_when_baseline_present(reports):
    _, over = reports
    md = cs._render_dataset_md(over)
    assert "## Distribution vs the baseline simulator" in md
    assert "LBGI (hypo risk)" in md


def test_distribution_is_deterministic(tmp_path):
    r1 = _build(tmp_path, "d1", hypo_oversample=0.5)
    r2 = _build(tmp_path, "d2", hypo_oversample=0.5)
    assert r1['distribution']['mean'] == r2['distribution']['mean']
    assert r1['distribution']['median'] == r2['distribution']['median']
    assert r1['distribution']['lbgi'] == r2['distribution']['lbgi']
