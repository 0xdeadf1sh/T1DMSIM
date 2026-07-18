"""
Compatibility test for the flat top-level meta.json keys the T1DMAI loader
requires.

T1DMAI's T1DMDataset._load_cache (data.py) reads a set of FLAT top-level keys
off meta.json and validates each. cache_simulator.py stores the same values
nested under report['params'] for DATASET.md, but _finalize_report also mirrors
them flat at the root so the ML-side loader can consume this cache unchanged.
This test builds a tiny cache and asserts the on-disk meta.json carries those
flat keys with the exact values the loader checks for.
"""

import json
import os
import sys

import pytest

pytest.importorskip("blosc2")  # cache_simulator imports blosc2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cache_simulator as cs  # noqa: E402


def test_meta_carries_flat_compat_keys(tmp_path):
    out_dir = str(tmp_path / "cache")
    cs.build_cache(
        out_dir=out_dir,
        pool_size=6,
        sim_hours=55.5,
        warmup_hours=48.0,
        n_jobs=1,
        dataset_md='',
        baseline_stats=None,
        force=True,
    )

    with open(os.path.join(out_dir, "meta.json")) as f:
        meta = json.load(f)

    required = (
        'pool_size', 'n_timesteps', 'sim_hours', 'simulator_warmup_hours',
        'patient_uniform_sample_prob', 'dt_minutes', 'channels', 'cache_format',
    )
    for key in required:
        assert key in meta, f"meta.json missing flat key {key!r}"

    assert meta['pool_size'] == 6
    assert meta['n_timesteps'] == 666
    assert meta['sim_hours'] == 55.5
    assert meta['simulator_warmup_hours'] == 48.0
    assert meta['patient_uniform_sample_prob'] == 0.0
    assert meta['dt_minutes'] == 5
    assert tuple(meta['channels']) == cs.CHANNEL_NAMES
    assert meta['cache_format'] == 'blosc2-ndarray-v1'
