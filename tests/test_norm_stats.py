"""
Tests for the normalization_stats.json the cache builder emits for T1DMAI.

cache_simulator.py writes, alongside the .b2nd channels + meta.json + DATASET.md,
a normalization_stats.json holding the 3-channel {mean, std} the downstream
T1DMAI model consumes: bg_absolute fit in Kovatchev risk space, carb_intake /
insulin_combined fit in log1p space, pooled over all rows x timesteps during the
transcode pass. These tests verify the file's schema and that its streaming
power-sum stats match a direct recompute from the stored .b2nd arrays.
"""

import json
import os
import sys

import numpy as np
import pytest

pytest.importorskip("blosc2")  # cache_simulator imports blosc2
import blosc2  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cache_simulator as cs  # noqa: E402


NORM_KEYS = ("bg_absolute", "carb_intake", "insulin_combined")


# ---------------------------------------------------------------------------
# Pure-function unit tests (fast, no simulation)
# ---------------------------------------------------------------------------

def test_norm_bg_risk_matches_reference():
    from simulator import BG_CLAMP_MIN, BG_CLAMP_MAX
    rng = np.random.default_rng(0)
    bg = rng.uniform(30, 450, size=5000)  # spans the clamp rails both ways
    got = cs._norm_bg_risk(bg)
    g = np.clip(bg.astype(np.float64), BG_CLAMP_MIN, BG_CLAMP_MAX)
    ref = cs.NORM_BG_RISK_SCALE * (np.log(g) ** cs.NORM_BG_RISK_POWER - cs.NORM_BG_RISK_OFFSET)
    assert np.allclose(got, ref, atol=1e-12)
    # Re-anchored endpoints: f(40) = -sqrt(10), f(400) = +sqrt(10).
    assert cs._norm_bg_risk(np.array([BG_CLAMP_MIN]))[0] == pytest.approx(-np.sqrt(10.0), abs=1e-6)
    assert cs._norm_bg_risk(np.array([BG_CLAMP_MAX]))[0] == pytest.approx(+np.sqrt(10.0), abs=1e-6)


def test_norm_bg_risk_is_not_clinical_kovatchev():
    # The clinical LBGI/HBGI constants (1.509 / 5.381) must stay untouched and
    # are distinct from the re-anchored model risk-space constants.
    assert cs.NORM_BG_RISK_SCALE != 1.509
    assert cs.NORM_BG_RISK_OFFSET != 5.381
    f = 1.509 * (np.log(120.0) ** 1.084 - 5.381)
    assert cs._norm_bg_risk(np.array([120.0]))[0] != pytest.approx(f, abs=1e-6)


def test_finalize_norm_stats_sample_std():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    acc = {"bg_absolute": {"sum": float(x.sum()),
                           "sum_sq": float((x * x).sum()), "n": x.size}}
    d = cs._finalize_norm_stats(acc)
    assert d["bg_absolute"]["mean"] == pytest.approx(x.mean())
    assert d["bg_absolute"]["std"] == pytest.approx(np.std(x, ddof=1))  # sample std


# ---------------------------------------------------------------------------
# End-to-end build test (tiny pool)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def built_cache(tmp_path_factory):
    out = tmp_path_factory.mktemp("caches") / "normcache"
    cs.build_cache(
        out_dir=str(out), pool_size=8, sim_hours=10.0, warmup_hours=5.0,
        n_jobs=1, dataset_md="", baseline_stats=None, seed_salt=13,
    )
    return str(out)


def test_norm_stats_file_schema(built_cache):
    path = os.path.join(built_cache, cs.NORM_STATS_FILE)
    assert os.path.exists(path), "normalization_stats.json must be emitted"
    with open(path) as f:
        stats = json.load(f)
    assert set(stats.keys()) == set(NORM_KEYS), "exactly the 3 T1DMAI channels"
    for name in NORM_KEYS:
        mean, std = stats[name]["mean"], stats[name]["std"]
        assert np.isfinite(mean) and np.isfinite(std)
        assert std > 0.0, f"{name} std must be positive"


def test_norm_stats_match_direct_recompute(built_cache):
    with open(os.path.join(built_cache, cs.NORM_STATS_FILE)) as f:
        stats = json.load(f)
    for cache_name, norm_name in cs.NORM_CHANNEL_SOURCES.items():
        arr = np.asarray(blosc2.open(os.path.join(built_cache, f"{cache_name}.b2nd"), mode="r"))
        tv = cs._norm_forward(cache_name, arr)  # same forward transform the builder uses
        assert stats[norm_name]["mean"] == pytest.approx(float(tv.mean()), rel=1e-6, abs=1e-6)
        assert stats[norm_name]["std"] == pytest.approx(float(np.std(tv, ddof=1)), rel=1e-6, abs=1e-6)


def test_norm_stats_deterministic(tmp_path):
    kw = dict(pool_size=8, sim_hours=10.0, warmup_hours=5.0, n_jobs=1,
              dataset_md="", baseline_stats=None, seed_salt=21)
    cs.build_cache(out_dir=str(tmp_path / "a"), **kw)
    cs.build_cache(out_dir=str(tmp_path / "b"), **kw)
    with open(os.path.join(str(tmp_path / "a"), cs.NORM_STATS_FILE)) as f:
        a = json.load(f)
    with open(os.path.join(str(tmp_path / "b"), cs.NORM_STATS_FILE)) as f:
        b = json.load(f)
    assert a == b
