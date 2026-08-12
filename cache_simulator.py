"""
cache_simulator.py -- pre-generate a pool of simulator trajectories to disk.
============================================================================

``T1DMSimulator.generate()`` is a stateful Python step loop; running it inside
every training DataLoader worker starves a fast GPU. This tool generates a pool
of ``N`` post-warmup trajectories once, compresses them to disk, and emits a
full statistical summary (``DATASET.md``) describing the corpus. Downstream
training then memory-maps the cache and never runs the simulator.

The script is deliberately dependency-light: ``numpy`` for numerics and
``blosc2`` for the compressed on-disk arrays. It never imports or mutates the
simulator's internals -- it only drives the public ``generate()`` API.

Layout
------
The output directory holds one compressed ``.b2nd`` (blosc2 NDArray) file per
channel, a raw ``icr.npy``, a ``normalization_stats.json``, and a ``meta.json``
sentinel::

    <out_dir>/
        bg_observed.b2nd        (N, T) float32   shuffle + zstd
        total_carb.b2nd         (N, T) float32   shuffle + zstd
        total_insulin.b2nd      (N, T) float32   shuffle + zstd
        insulin_resistance.b2nd (N, T) float32   shuffle + zstd
        hgo.b2nd                (N, T) float32   shuffle + zstd
        total_exercise.b2nd     (N, T) float32   shuffle + zstd
        hour_of_day.b2nd        (N, T) float32   shuffle + zstd
        day.b2nd                (N, T) int32     shuffle + zstd
        icr.npy                 (N,)   float32   raw (tiny)
        normalization_stats.json  3-channel {mean, std} the T1DMAI model consumes
        meta.json               generation params + aggregate statistics

``normalization_stats.json`` is the ``{bg_absolute, carb_intake,
insulin_combined}: {mean, std}`` contract the downstream T1DMAI model reads: the
bg mean/std are fit in Kovatchev risk space, carb/insulin in log1p space (the
same forward transforms the model applies before its z-score), pooled over all
rows x timesteps during the transcode pass. It is written durably before the
meta.json sentinel so a completed cache always carries valid stats.

``DATASET.md`` (path from ``--dataset-md``, default ``./DATASET.md``) is a
human-readable render of the same statistics.

Performance
-----------
Single-threaded: each patient calls ``generate()`` exactly ``warmup + keep``
times; the warmup steps are advanced and discarded, and only the eight cached
channels (plus the basal/bolus split used for statistics) are pulled out of the
per-step dict straight into pre-allocated arrays -- avoiding the 21-list build
that ``generate_hours`` performs.

Multi-threaded: the simulator is pure-Python and GIL-bound, so parallelism is
via processes (``fork`` on Linux -- the already-imported simulator module is
inherited, so there is no per-worker re-import). Each worker simulates one row
and writes it *directly* into a shared per-channel ``.npy`` staging memmap at
its own row offset; only a tiny per-patient statistics dict is returned over the
IPC boundary, never the (T,)-length channel arrays. Disjoint-row writes into a
``MAP_SHARED`` file-backed mapping are safe across processes, and the pool join
happens-before the transcode read.

After the worker fan-out, a single transcode pass converts each staging channel
to a compressed ``.b2nd`` (byte-shuffle + zstd), deleting the ``.npy`` as soon
as its channel finishes so peak disk stays near ``max(raw, compressed)`` rather
than their sum. Every file and the staging directory are fsync'd before
``meta.json`` (the sentinel) is written and the directory is atomically renamed
into place, so neither a process crash nor a power loss can surface a
sentinel-present, data-truncated cache.

blosc2 parallelizes compression across the blocks *within* a chunk, so transcode
throughput scales with ``--rows-per-chunk`` (a small chunk is one block and
transcodes single-threaded); that same value is the random-row read granularity,
so it trades one-time build speed against per-read decompression waste.

Window validity filter
----------------------
Any trajectory whose ``bg_observed`` touches the clamp rails -- a reading
``>= --rail-high`` or ``<= --rail-low``, both derived from BG_CLAMP_MAX/MIN --
is discarded and a fresh seed is drawn for that row. This keeps rows pinned flat
against a clamp out of the training pool. The low rail tracks the dynamics floor,
not the CGM reporting floor: it must never discard a trajectory merely for
reaching a severe low, which is a population the pool exists to carry.

Hypo oversampling
-----------------
With ``--hypo-oversample P``, a deterministic fraction ``P`` of rows are
"hypo-designated": for those rows the seed is re-rolled until the trajectory
spends at least ``--hypo-min-frac`` of its time below 70 mg/dL (or the attempt
cap is hit, in which case the hypo-richest attempt is kept). Every row -- hypo
or not -- still respects the rail filter, and each stored row is one whole,
physiologically-consistent patient. Rejection sampling is per-row and keyed on
``(seed_salt, cache_idx, attempt)``, so the pool is fully reproducible and the
work parallelizes without coordination.

Usage
-----
::

    python cache_simulator.py --out-dir simulator_cache --pool-size 50000
    python cache_simulator.py --pool-size 20000 --hypo-oversample 0.25 --jobs 30
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import json
import math
import multiprocessing as mp
import os
import shutil
import sys
import time
from typing import Any

import numpy as np

# Run-from-repo-root convenience: make ``import simulator`` resolve regardless
# of the caller's CWD.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import blosc2

from simulator import (BG_CLAMP_MIN, BG_CLAMP_MAX, DT_MINUTES,
                       SIMULATOR_WARMUP_HOURS, T1DMSimulator)


# ============================================================================
# CHANNELS
# ============================================================================

CHANNEL_NAMES = (
    'bg_observed',
    'total_carb',
    'total_insulin',
    'insulin_resistance',
    'hgo',
    'total_exercise',
    'hour_of_day',
    'day',
)
CHANNEL_DTYPES: dict[str, Any] = {
    'bg_observed': np.float32,
    'total_carb': np.float32,
    'total_insulin': np.float32,
    'insulin_resistance': np.float32,
    'hgo': np.float32,
    'total_exercise': np.float32,
    'hour_of_day': np.float32,
    'day': np.int32,
}

# ============================================================================
# CACHE / COMPRESSION PARAMETERS
# ============================================================================

CACHE_FORMAT_VERSION = 'blosc2-ndarray-v1'
DEFAULT_ROWS_PER_CHUNK = 32
DEFAULT_ZSTD_CLEVEL = 5
# blosc2 chunks must fit a signed 32-bit byte count (the C-extension ceiling).
_BLOSC2_MAX_CHUNK_BYTES = 2**31 - 1
# Transcode a staging channel in row-blocks of about this many bytes so a
# 266 GB/channel pool (100M patients) never has to fit a whole channel in RAM,
# while each block still spans many chunks for blosc2 to compress in parallel.
_TRANSCODE_BLOCK_BYTES = 256 * 1024 * 1024

# ============================================================================
# GENERATION DEFAULTS
# ============================================================================

DEFAULT_SIM_HOURS = 103.5    # [CF] 55.5→103.5 — room for a 72 h context, at the only nearby length that keeps hour-of-day coverage exactly uniform. T1DMAI draws patch-aligned starts in [context, N - forward_room], so the start count is (N - context - forward)/6 + 1 and must be a multiple of 48 (half-hour slots per day) or some slots are oversampled. N = 1242 gives 144 starts under today's 24 h context and 48 under a 72 h one; 1248 gives 145 and 49, which tilts every window distribution toward midnight. PAIRED CHANGE: T1DMAI/data.py ON_THE_FLY_SIM_HOURS must move with this or its loader rejects the cache outright.
DEFAULT_WARMUP_HOURS = SIMULATOR_WARMUP_HOURS  # discarded lead-in so the state has forgotten its init
DEFAULT_MAX_ATTEMPTS = 50    # soft cap: budget for meeting the hypo target per row
# Extra draws devoted purely to finding a rail-VALID trajectory once the hypo
# budget is spent, before giving up and raising. Validity is a hard requirement,
# so this backstops the (essentially unreachable) case of a run of rail-hitters.
_VALIDITY_HARD_EXTRA = 250
DEFAULT_HYPO_MIN_FRAC = 0.05 # min fraction of time < 70 mg/dL for a hypo-designated row
DEFAULT_RAIL_HIGH = BG_CLAMP_MAX - 1.0  # discard a window with any bg_observed >= this
DEFAULT_RAIL_LOW = BG_CLAMP_MIN + 1.0   # [CF] was a hardcoded 41, which discarded every trajectory reaching a severe low — precisely the population this corpus now exists to teach. Both rails are DERIVED so a future clamp change cannot desynchronise them; the filter only ever meant to drop rows pinned flat against a clamp.

# Event-detection thresholds (mirrors the README's meal detector: upward
# threshold-crossings with a refractory window).
MEAL_CARB_THRESHOLD_G = 1.0       # a meal step appears > 1 g/step of carb
BOLUS_EVENT_THRESHOLD_U = 0.02    # a bolus onset crosses this U/step
EVENT_REFRACTORY_MIN = 30.0       # min gap between counted events

# Glycemic-band edges (mg/dL) for the temporal class ratios and TIR/TBR/TAR.
BG_SEVERE_LOW = 54.0
BG_LOW = 70.0
BG_HIGH = 180.0
BG_VERY_HIGH = 250.0

# 1 mg/dL histogram over the CGM clamp span, accumulated during the transcode
# pass to recover pooled percentiles/median without holding every reading in RAM.
BG_HIST_EDGES = np.arange(BG_CLAMP_MIN, BG_CLAMP_MAX + 1.0, 1.0)  # derived, never a literal: a hardcoded 40 floor silently drops every sub-40 reading from the pooled percentiles while still counting it in n

# Path (relative to this file) to the diff report's machine-readable baseline.
DEFAULT_BASELINE_STATS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'diff', 'stats.json'
)


def _kovatchev_numerators(bg: np.ndarray) -> tuple[float, float]:
    """Sum of the Kovatchev (1997) LBGI/HBGI per-reading risk terms over ``bg``.

    Matches ``diff/build_report.kovatchev_risk``: f = 1.509*(ln(bg)^1.084 - 5.381),
    left/hypo risk 10*min(f,0)^2, right/hyper risk 10*max(f,0)^2. Returns the
    *sums* so the caller divides by the pooled reading count to get LBGI/HBGI.
    """
    # Clinical domain floor is 20 mg/dL (SPEC/invariants.md, and T1DMDROID's
    # CLINICAL_BG_CLAMP_MIN). The old 1.0 guard was merely an ln() safety net and
    # was unreachable while BG could not fall below 40; it is reachable now.
    f = 1.509 * (np.log(np.clip(bg, 20.0, None)) ** 1.084 - 5.381)
    rl = 10.0 * np.minimum(f, 0.0) ** 2
    rh = 10.0 * np.maximum(f, 0.0) ** 2
    return float(rl.sum(dtype=np.float64)), float(rh.sum(dtype=np.float64))


# ============================================================================
# NORMALIZATION STATS (consumed by the T1DMAI model; self-contained -- no T1DMAI
# import). Emits normalization_stats.json alongside meta.json.
# ============================================================================

# Kovatchev risk constants anchored so f(40) = -sqrt(10) and f(400) = +sqrt(10);
# mirror T1DMAI/utils._KOVATCHEV_* -- model risk space, distinct from the clinical
# LBGI/HBGI constants used by _kovatchev_numerators. The anchor points are a
# property of these constants and stay fixed; they are NOT tied to
# BG_CLAMP_MIN/MAX, which now clip to [10, 400] so the transform extends to
# f(10) = -6.82. Re-anchoring to the new floor would compress the hypoglycemic
# stretch, which is the whole reason the model forecasts in this space.
NORM_BG_RISK_SCALE = 2.2211457449985317
NORM_BG_RISK_POWER = 1.084
NORM_BG_RISK_OFFSET = 5.540076976170212

# File name of the emitted 3-channel {mean, std} stats the T1DMAI model reads.
NORM_STATS_FILE = 'normalization_stats.json'

# Cache channel -> T1DMAI normalization channel name. bg_absolute is fit in
# Kovatchev risk space, carb_intake / insulin_combined in log1p space (see
# _norm_forward); the T1DMAI model applies the same forward transform before its
# z-score, so the fitted mean/std live in those spaces.
NORM_CHANNEL_SOURCES: dict[str, str] = {
    'bg_observed': 'bg_absolute',
    'total_carb': 'carb_intake',
    'total_insulin': 'insulin_combined',
}


def _norm_bg_risk(bg: np.ndarray) -> np.ndarray:
    """Kovatchev risk transform (model risk space) of raw mg/dL ``bg``.

    Clips to ``[BG_CLAMP_MIN, BG_CLAMP_MAX]`` then applies
    ``f(g) = SCALE * (ln(g)^POWER - OFFSET)`` with the constants above, anchored
    at ``f(40) = -sqrt(10)`` / ``f(400) = +sqrt(10)`` -- mirroring
    ``T1DMAI/utils.kovatchev_f_np``, the transform the model applies to the bg
    input BEFORE the z-score. The clip floor sits below the lower anchor, so the
    output range is asymmetric: ``[f(10), f(400)] = [-6.82, +3.16]``.
    Returns float64.
    """
    g = np.clip(np.asarray(bg, dtype=np.float64), BG_CLAMP_MIN, BG_CLAMP_MAX)
    return NORM_BG_RISK_SCALE * (np.log(g) ** NORM_BG_RISK_POWER - NORM_BG_RISK_OFFSET)


def _norm_forward(cache_name: str, block: np.ndarray) -> np.ndarray:
    """Forward transform for the normalization-stats fit of ``cache_name``.

    ``bg_observed`` -> Kovatchev risk space; ``total_carb`` / ``total_insulin``
    -> ``log1p(max(x, 0))``. Returns the flattened float64 transformed values.
    """
    v = np.asarray(block, dtype=np.float64).reshape(-1)
    if cache_name == 'bg_observed':
        return _norm_bg_risk(v)
    return np.log1p(np.maximum(v, 0.0))


def _finalize_norm_stats(
    norm_acc: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Reduce the power-sum accumulators to the T1DMAI ``{name: {mean, std}}`` contract.

    ``std`` is the SAMPLE std ``sqrt(M2 / (n - 1))`` (``M2 = sum_sq - sum^2/n``),
    matching ``T1DMAI/normalization._finalize_stats``. Returns exactly the three
    channels bg_absolute / carb_intake / insulin_combined.
    """
    stats: dict[str, dict[str, float]] = {}
    for name, a in norm_acc.items():
        n = a['n']
        mean = a['sum'] / n if n else 0.0
        m2 = (a['sum_sq'] - (a['sum'] * a['sum']) / n) if n else 0.0
        std = float(np.sqrt(max(m2, 0.0) / max(n - 1, 1)))
        stats[name] = {'mean': float(mean), 'std': std}
    return stats


@dataclasses.dataclass(frozen=True)
class RowConfig:
    """Per-row rejection-sampling knobs, shared read-only across workers."""
    warmup_steps: int
    keep_steps: int
    max_attempts: int
    rail_high: float
    rail_low: float
    hypo_prob: float
    hypo_min_frac: float
    hypo_threshold: float
    seed_salt: int
    event_refractory_steps: int


# ============================================================================
# STEP-COUNT / SEED HELPERS
# ============================================================================

def _steps_for_hours(hours: float) -> int:
    """Number of 5-minute steps in ``hours`` (floored, matching the simulator)."""
    return int(hours * 60 / DT_MINUTES)


def _row_seed(cache_idx: int, attempt: int, seed_salt: int) -> int:
    """
    Deterministic 63-bit simulator seed for ``(cache_idx, attempt)``.

    Keyed on ``seed_salt`` so different salts draw independent corpora, and on
    ``attempt`` so a rejected trajectory's re-roll is reproducible.
    """
    digest = hashlib.sha256(f'{seed_salt}:{cache_idx}:{attempt}'.encode()).hexdigest()
    return int(digest, 16) % (2**63 - 1)


def _is_hypo_row(cache_idx: int, prob: float, seed_salt: int) -> bool:
    """Deterministic hypo-designation decision for ``cache_idx`` (uniform < prob)."""
    if prob <= 0.0:
        return False
    digest = hashlib.sha256(f'hypo:{seed_salt}:{cache_idx}'.encode()).hexdigest()
    u = (int(digest, 16) % 1_000_000_000) / 1_000_000_000
    return u < prob


def _count_crossings(x: np.ndarray, thresh: float, refractory_steps: int) -> int:
    """
    Count upward threshold-crossings of ``x`` above ``thresh`` with a refractory.

    A crossing is a step where ``x`` rises from ``<= thresh`` to ``> thresh``;
    crossings closer than ``refractory_steps`` to the previous counted one are
    merged. A signal already above ``thresh`` at index 0 is not counted (no
    upward crossing is observed), matching the README's meal detector.
    """
    above = x > thresh
    if above.size < 2:
        return int(above.any())
    rising = np.flatnonzero(above[1:] & ~above[:-1]) + 1
    if rising.size == 0:
        return 0
    count = 0
    last = -(10**9)
    for idx in rising:
        idx = int(idx)
        if idx - last >= refractory_steps:
            count += 1
            last = idx
    return count


# ============================================================================
# SINGLE-PATIENT SIMULATION + STATISTICS
# ============================================================================

def _simulate_patient(
    seed: int, cfg: RowConfig
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """
    Simulate one patient and return its cached channels plus a statistics dict.

    Advances ``cfg.warmup_steps`` steps unstored, then pulls the eight cached
    channels (and the basal/bolus split, used only for statistics) out of the
    per-step ``generate()`` dict into pre-allocated arrays.
    """
    sim = T1DMSimulator(seed=int(seed))
    gen = sim.generate

    for _ in range(cfg.warmup_steps):
        gen()

    T = cfg.keep_steps
    bg_obs = np.empty(T, np.float32)
    carb = np.empty(T, np.float32)
    ins = np.empty(T, np.float32)
    ir = np.empty(T, np.float32)
    hgo = np.empty(T, np.float32)
    exer = np.empty(T, np.float32)
    hod = np.empty(T, np.float32)
    day = np.empty(T, np.int32)
    basal = np.empty(T, np.float32)   # not cached; only for the basal:bolus split
    bolus = np.empty(T, np.float32)

    for i in range(T):
        s = gen()
        bg_obs[i] = s['bg_observed']
        carb[i] = s['total_carb']
        ins[i] = s['total_insulin']
        ir[i] = s['insulin_resistance']
        hgo[i] = s['hgo']
        exer[i] = s['total_exercise']
        hod[i] = s['hour_of_day']
        day[i] = s['day']
        basal[i] = s['basal_insulin']
        bolus[i] = s['bolus_insulin']

    channels = {
        'bg_observed': bg_obs,
        'total_carb': carb,
        'total_insulin': ins,
        'insulin_resistance': ir,
        'hgo': hgo,
        'total_exercise': exer,
        'hour_of_day': hod,
        'day': day,
    }

    ref = cfg.event_refractory_steps
    below54 = int((bg_obs < BG_SEVERE_LOW).sum())
    below70 = int((bg_obs < BG_LOW).sum())
    above180 = int((bg_obs > BG_HIGH).sum())
    above250 = int((bg_obs > BG_VERY_HIGH).sum())
    in_range = T - below70 - above180

    stats: dict[str, Any] = {
        'icr': float(sim.patient.icr),
        'sum_carb': float(carb.sum(dtype=np.float64)),
        'sum_insulin': float(ins.sum(dtype=np.float64)),
        'sum_basal': float(basal.sum(dtype=np.float64)),
        'sum_bolus': float(bolus.sum(dtype=np.float64)),
        'sum_bg': float(bg_obs.sum(dtype=np.float64)),
        'bg_min': float(bg_obs.min()),
        'bg_max': float(bg_obs.max()),
        'below54': below54,
        'below70': below70,
        'in_range': in_range,
        'above180': above180,
        'above250': above250,
        'n_meals': _count_crossings(carb, MEAL_CARB_THRESHOLD_G, ref),
        'n_boluses': _count_crossings(bolus, BOLUS_EVENT_THRESHOLD_U, ref),
        'n_ex_sessions': _count_crossings(exer, 0.0, ref),
        'ex_active_steps': int((exer > 0.0).sum()),
        # Fraction of time below the configurable hypo cutoff, used only for the
        # oversampling criterion (the class ratios below always use the fixed 70).
        'frac_below_hypo': float((bg_obs < cfg.hypo_threshold).mean()),
    }
    return channels, stats


def _finish(
    channels: dict[str, np.ndarray], stats: dict[str, Any],
    cache_idx: int, n_sims: int, is_hypo: bool, accepted: bool,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    stats['idx'] = cache_idx
    stats['attempts'] = n_sims
    stats['accepted'] = accepted
    stats['is_hypo_row'] = is_hypo
    return channels, stats


def _resolve_row(cache_idx: int, cfg: RowConfig) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """
    Rejection-sample the trajectory stored at ``cache_idx``.

    Rail-validity (no bg_observed touching the clamp rails) is a HARD gate: an
    invalid trajectory is never a stored candidate. Seeds are redrawn until a
    valid one is found, and -- for a hypo-designated row -- a valid trajectory
    that also spends at least ``hypo_min_frac`` of its time below
    ``hypo_threshold`` is preferred. The hypo target is the only *soft* one: if
    it cannot be met within ``max_attempts``, the hypo-richest *valid* trajectory
    is kept (``accepted=False``). Validity is pursued for up to
    ``max_attempts + _VALIDITY_HARD_EXTRA`` draws before giving up; exhausting
    that without a single valid trajectory raises rather than store a rail-hitter
    (only reachable with a degenerate rail band).
    """
    is_hypo = _is_hypo_row(cache_idx, cfg.hypo_prob, cfg.seed_salt)
    hard_cap = cfg.max_attempts + _VALIDITY_HARD_EXTRA
    best_valid: tuple[dict[str, np.ndarray], dict[str, Any]] | None = None
    best_hypo = -1.0
    n_sims = 0

    for attempt in range(hard_cap):
        seed = _row_seed(cache_idx, attempt, cfg.seed_salt)
        channels, stats = _simulate_patient(seed, cfg)
        n_sims += 1

        if not (stats['bg_max'] < cfg.rail_high and stats['bg_min'] > cfg.rail_low):
            continue  # rail-hitting: never a stored candidate

        if not is_hypo:
            return _finish(channels, stats, cache_idx, n_sims, is_hypo, accepted=True)
        if stats['frac_below_hypo'] >= cfg.hypo_min_frac:
            return _finish(channels, stats, cache_idx, n_sims, is_hypo, accepted=True)

        # Valid but hypo target unmet: remember the hypo-richest valid attempt and
        # keep spending the (soft) hypo budget; once it is exhausted, keep the best.
        if stats['frac_below_hypo'] > best_hypo:
            best_hypo = stats['frac_below_hypo']
            best_valid = (channels, stats)
        if attempt + 1 >= cfg.max_attempts and best_valid is not None:
            ch, st = best_valid
            return _finish(ch, st, cache_idx, n_sims, is_hypo, accepted=False)

    if best_valid is not None:
        ch, st = best_valid
        return _finish(ch, st, cache_idx, n_sims, is_hypo, accepted=False)
    raise RuntimeError(
        f'row {cache_idx}: no rail-valid trajectory (bg_observed in '
        f'({cfg.rail_low}, {cfg.rail_high})) found in {hard_cap} attempts; '
        f'check --rail-low / --rail-high.'
    )


# ============================================================================
# WORKER MACHINERY
# ============================================================================

_WORKER: dict[str, Any] = {}


def _worker_init(partial_dir: str, cfg: RowConfig) -> None:
    """Open the per-channel staging memmaps (r+) and stash the row config."""
    _WORKER['cfg'] = cfg
    _WORKER['mm'] = {
        name: np.lib.format.open_memmap(os.path.join(partial_dir, f'{name}.npy'), mode='r+')
        for name in CHANNEL_NAMES
    }


def _worker(cache_idx: int) -> dict[str, Any]:
    """Resolve one row, write it straight into the staging memmaps, return stats."""
    cfg: RowConfig = _WORKER['cfg']
    mm = _WORKER['mm']
    channels, stats = _resolve_row(cache_idx, cfg)
    for name in CHANNEL_NAMES:
        mm[name][cache_idx] = channels[name]
    return stats


# ============================================================================
# STREAMING STATISTICS ACCUMULATOR
# ============================================================================

class StatAccumulator:
    """Folds per-patient statistics into running aggregates (O(1) memory)."""

    # Folded in worker-completion order, so the float sums (sum_carb/…/sum_bg)
    # may differ by trailing ULPs run-to-run; the cached channels and icr are
    # bit-identical, and DATASET.md rounds these, so this is cosmetic only.
    _SUM_KEYS = (
        'sum_carb', 'sum_insulin', 'sum_basal', 'sum_bolus', 'sum_bg',
        'below54', 'below70', 'in_range', 'above180', 'above250',
        'n_meals', 'n_boluses', 'n_ex_sessions', 'ex_active_steps',
    )

    def __init__(self) -> None:
        self.n_rows = 0
        self.sums = {k: 0.0 for k in self._SUM_KEYS}
        self.icr_sum = 0.0
        self.icr_min = math.inf
        self.icr_max = -math.inf
        self.bg_min = math.inf
        self.bg_max = -math.inf
        self.total_attempts = 0
        self.n_fallback = 0
        self.n_hypo_rows = 0
        self.n_hypo_qualified = 0

    def add(self, stats: dict[str, Any]) -> None:
        self.n_rows += 1
        for k in self._SUM_KEYS:
            self.sums[k] += stats[k]
        icr = stats['icr']
        self.icr_sum += icr
        self.icr_min = min(self.icr_min, icr)
        self.icr_max = max(self.icr_max, icr)
        self.bg_min = min(self.bg_min, stats['bg_min'])
        self.bg_max = max(self.bg_max, stats['bg_max'])
        self.total_attempts += stats['attempts']
        if not stats['accepted']:
            self.n_fallback += 1
        if stats['is_hypo_row']:
            self.n_hypo_rows += 1
            if stats['accepted']:
                self.n_hypo_qualified += 1


# ============================================================================
# TRANSCODE: staging .npy -> compressed .b2nd
# ============================================================================

def _resolve_rows_per_chunk(rows_per_chunk: int, pool_size: int, n_timesteps: int) -> int:
    """Clamp ``rows_per_chunk`` so a chunk fits blosc2's 2^31-1 byte ceiling and <= pool_size."""
    if rows_per_chunk < 1:
        raise ValueError(f'rows_per_chunk must be >= 1, got {rows_per_chunk}')
    max_itemsize = max(np.dtype(d).itemsize for d in CHANNEL_DTYPES.values())
    row_bytes = n_timesteps * max_itemsize
    if row_bytes > _BLOSC2_MAX_CHUNK_BYTES:
        raise ValueError(
            f'n_timesteps={n_timesteps} x itemsize={max_itemsize} = {row_bytes} '
            f'bytes/row exceeds blosc2 chunk limit {_BLOSC2_MAX_CHUNK_BYTES}'
        )
    byte_cap = _BLOSC2_MAX_CHUNK_BYTES // row_bytes
    return int(min(rows_per_chunk, pool_size, byte_cap))


def _fsync_file(path: str) -> None:
    """Flush a file's data to disk (best-effort; durability, not correctness)."""
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def _fsync_dir(path: str) -> None:
    """Flush a directory entry to disk so a rename/create survives power loss."""
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def _transcode_to_blosc2(
    partial_dir: str, pool_size: int, n_timesteps: int,
    rows_per_chunk: int, clevel: int, nthreads: int,
) -> tuple[dict[str, dict[str, int]], dict[str, Any], dict[str, dict[str, float]]]:
    """
    Convert each staging ``.npy`` to a compressed ``.b2nd`` and delete the source.

    Returns ``(inventory, bg_dist, norm_acc)``: a per-channel
    ``{raw_bytes, compressed_bytes}`` inventory; the pooled ``bg_observed``
    distribution aggregates (histogram, power sums, Kovatchev numerators); and the
    per-channel ``{sum, sum_sq, n}`` power sums of the T1DMAI forward-transformed
    bg/carb/insulin channels (risk space for bg, log1p for carb/insulin) for the
    ``normalization_stats.json`` fit. All are accumulated block-by-block as each
    channel is read for compression -- so pooled percentiles / std / skew / LBGI /
    HBGI and the normalization stats cost no extra I/O and never materialise a
    whole channel in RAM. Round-trips the first and last row of each channel to
    catch chunk-edge corruption.
    """
    bg_hist = np.zeros(len(BG_HIST_EDGES) - 1, dtype=np.float64)
    bg_dist = {'sum': 0.0, 'sum_sq': 0.0, 'sum_cube': 0.0,
               'sum_rl': 0.0, 'sum_rh': 0.0, 'n': 0}
    norm_acc: dict[str, dict[str, float]] = {
        norm: {'sum': 0.0, 'sum_sq': 0.0, 'n': 0}
        for norm in NORM_CHANNEL_SOURCES.values()
    }
    cparams = blosc2.CParams(
        codec=blosc2.Codec.ZSTD,
        clevel=clevel,
        filters=[blosc2.Filter.SHUFFLE],
        nthreads=max(1, nthreads),
    )
    max_itemsize = max(np.dtype(d).itemsize for d in CHANNEL_DTYPES.values())
    # Round the RAM-bounded block down to a whole number of chunks (>= 1 chunk)
    # so block edges land on chunk boundaries and blosc2 recompresses nothing.
    rows_per_block = max(
        rows_per_chunk,
        (_TRANSCODE_BLOCK_BYTES // (n_timesteps * max_itemsize) // rows_per_chunk) * rows_per_chunk,
    )
    inventory: dict[str, dict[str, int]] = {}

    for name in CHANNEL_NAMES:
        src_path = os.path.join(partial_dir, f'{name}.npy')
        dst_path = os.path.join(partial_dir, f'{name}.b2nd')
        dtype = CHANNEL_DTYPES[name]

        src = np.load(src_path, mmap_mode='r')
        assert src.shape == (pool_size, n_timesteps), (
            f'staged {name!r} has shape {src.shape}, expected {(pool_size, n_timesteps)}'
        )

        dst = None
        reopened = None
        try:
            dst = blosc2.empty(
                shape=(pool_size, n_timesteps),
                chunks=(rows_per_chunk, n_timesteps),
                dtype=dtype,
                urlpath=dst_path,
                mode='w',
                cparams=cparams,
            )
            # Block-wise assignment bounds peak RAM at one block rather than a
            # whole channel. blosc2 parallelizes compression across the *blocks
            # within a chunk*, so throughput scales with rows_per_chunk (a small
            # chunk is a single block and transcodes single-threaded regardless
            # of nthreads) -- see the rows_per_chunk help for the read tradeoff.
            for r0 in range(0, pool_size, rows_per_block):
                r1 = min(r0 + rows_per_block, pool_size)
                block = np.asarray(src[r0:r1])
                dst[r0:r1] = block
                if name == 'bg_observed':
                    v = block.reshape(-1).astype(np.float64)
                    bg_hist += np.histogram(v, bins=BG_HIST_EDGES)[0]
                    bg_dist['sum'] += float(v.sum())
                    bg_dist['sum_sq'] += float((v * v).sum())
                    bg_dist['sum_cube'] += float((v * v * v).sum())
                    rl, rh = _kovatchev_numerators(v)
                    bg_dist['sum_rl'] += rl
                    bg_dist['sum_rh'] += rh
                    bg_dist['n'] += int(v.size)
                if name in NORM_CHANNEL_SOURCES:
                    tv = _norm_forward(name, block)
                    na = norm_acc[NORM_CHANNEL_SOURCES[name]]
                    na['sum'] += float(tv.sum())
                    na['sum_sq'] += float((tv * tv).sum())
                    na['n'] += int(tv.size)

            reopened = blosc2.open(dst_path, mode='r')
            for check_idx in (0, pool_size - 1):
                got = np.asarray(reopened[check_idx:check_idx + 1])[0]
                if not np.array_equal(got, np.asarray(src[check_idx])):
                    raise RuntimeError(
                        f'transcode round-trip mismatch on {name!r} row {check_idx}'
                    )
        finally:
            del reopened
            del dst
            gc.collect()

        _fsync_file(dst_path)  # durably land the frame before its .npy is unlinked
        raw_bytes = int(src.nbytes)
        del src
        os.remove(src_path)
        cbytes = int(os.path.getsize(dst_path))
        inventory[name] = {'raw_bytes': raw_bytes, 'compressed_bytes': cbytes}
        ratio = raw_bytes / max(cbytes, 1)
        print(
            f'  transcoded {name:<20} {raw_bytes / 1e6:8.2f} MB -> '
            f'{cbytes / 1e6:8.2f} MB ({ratio:.2f}x)',
            flush=True,
        )
    bg_dist['hist'] = bg_hist.tolist()
    return inventory, bg_dist, norm_acc


# ============================================================================
# REPORT ASSEMBLY + RENDERING
# ============================================================================

def _hist_percentile(hist: np.ndarray, edges: np.ndarray, q: float) -> float:
    """Linear-interpolated q-th percentile (0-100) from a histogram."""
    total = float(hist.sum())
    if total <= 0:
        return float('nan')
    target = (q / 100.0) * total
    cum = np.cumsum(hist)
    idx = int(np.searchsorted(cum, target, side='left'))
    idx = min(idx, len(hist) - 1)
    below = float(cum[idx - 1]) if idx > 0 else 0.0
    within = (target - below) / hist[idx] if hist[idx] > 0 else 0.0
    lo, hi = float(edges[idx]), float(edges[idx + 1])
    return lo + within * (hi - lo)


def _pool_distribution(bg_dist: dict[str, Any]) -> dict[str, float]:
    """Pooled bg_observed moments / percentiles / LBGI-HBGI from the transcode aggregates."""
    n = bg_dist['n']
    if not n:
        return {}
    hist = np.asarray(bg_dist['hist'], dtype=np.float64)
    mean = bg_dist['sum'] / n
    var = max(bg_dist['sum_sq'] / n - mean * mean, 0.0)
    std = var ** 0.5
    m3 = bg_dist['sum_cube'] / n - 3 * mean * (bg_dist['sum_sq'] / n) + 2 * mean ** 3
    skew = m3 / (std ** 3) if std > 0 else 0.0
    p = {q: _hist_percentile(hist, BG_HIST_EDGES, q) for q in (5, 25, 50, 75, 95)}
    return {
        'mean': mean, 'std': std, 'cv_pct': 100.0 * std / mean if mean else 0.0,
        'skew': skew, 'median': p[50],
        'p5': p[5], 'p25': p[25], 'p75': p[75], 'p95': p[95],
        'lbgi': bg_dist['sum_rl'] / n, 'hbgi': bg_dist['sum_rh'] / n,
    }


def _load_baseline_stats(path: str) -> dict[str, Any] | None:
    """
    Load the unbiased-simulator baseline distribution from the diff report's
    ``stats.json`` (``datasets.Sim``). Returns None if the file is missing or its
    schema does not match, so the comparison section is simply omitted.
    """
    try:
        with open(path) as f:
            sim = json.load(f)['datasets']['Sim']
        pm, pp, pr, sm = (sim['pooled_moments'], sim['pooled_percentiles'],
                          sim['pooled_risk'], sim['summary'])

        def band(key: str) -> float:
            return sm[key]['mean'] / 100.0

        return {
            'source': path,
            'n_records': sim.get('n_records'),
            'mean': pm['mean'], 'std': pm['std'], 'median': pm['median'],
            'cv_pct': pm['cv_pct'], 'skew': pm['skew'],
            'p5': pp['p5'], 'p25': pp['p25'], 'p75': pp['p75'], 'p95': pp['p95'],
            'lbgi': pr['LBGI_pooled'], 'hbgi': pr['HBGI_pooled'],
            'frac_below54': band('TBR2_pct'),
            'frac_below70': band('TBR1_pct') + band('TBR2_pct'),
            'frac_in_range': band('TIR_pct'),
            'frac_above180': band('TAR1_pct') + band('TAR2_pct'),
            'frac_above250': band('TAR2_pct'),
        }
    except (OSError, KeyError, ValueError, TypeError):
        return None


def _finalize_report(
    acc: StatAccumulator,
    params: dict[str, Any],
    inventory: dict[str, dict[str, int]],
    icr_bytes: int,
    bg_dist: dict[str, Any],
    baseline: dict[str, Any] | None,
) -> dict[str, Any]:
    """Reduce the accumulator + inventory into the numbers DATASET.md renders."""
    n = acc.n_rows
    T = params['n_timesteps']
    total_steps = n * T
    keep_hours = T * DT_MINUTES / 60.0
    days_per_patient = keep_hours / 24.0

    sums = acc.sums
    total_carb_g = sums['sum_carb']
    total_insulin_u = sums['sum_insulin']
    total_basal_u = sums['sum_basal']
    total_bolus_u = sums['sum_bolus']

    def frac(count: float) -> float:
        return count / total_steps if total_steps else 0.0

    compressed_total = sum(c['compressed_bytes'] for c in inventory.values())
    raw_total = sum(c['raw_bytes'] for c in inventory.values())
    # meta.json is a few KB (negligible vs the channels) and its size is
    # self-referential to record, so it is excluded from the footprint total.
    disk_total = compressed_total + icr_bytes

    return {
        'schema': 'dataset-report-v1',
        'params': params,
        # Flat top-level keys the T1DMAI T1DMDataset loader (data.py
        # _load_cache) hard-requires. They mirror values already nested under
        # 'params'; kept flat here because that loader reads them off the root
        # of meta.json. patient_uniform_sample_prob is hardcoded 0.0 because
        # this generator never mixes uniform-skill patients.
        'pool_size': params['pool_size'],
        'n_timesteps': params['n_timesteps'],
        'sim_hours': params['sim_hours'],
        'simulator_warmup_hours': params['warmup_hours'],
        'patient_uniform_sample_prob': 0.0,
        'dt_minutes': params['dt_minutes'],
        'channels': params['channels'],
        'cache_format': params['cache_format'],
        'rows_per_chunk': params['rows_per_chunk'],
        'zstd_clevel': params['zstd_clevel'],
        'corpus': {
            'n_patients': n,
            'n_timesteps': T,
            'dt_minutes': DT_MINUTES,
            'keep_hours': keep_hours,
            'days_per_patient': days_per_patient,
            'total_readings': total_steps,
            'total_cgm_hours': total_steps * DT_MINUTES / 60.0,
            'total_cgm_years': total_steps * DT_MINUTES / 60.0 / 8760.0,
        },
        'consumption': {
            'carb_total_g': total_carb_g,
            'carb_total_tonnes': total_carb_g / 1e6,
            'carb_per_patient_g': total_carb_g / n if n else 0.0,
            'carb_per_day_g': (total_carb_g / n / days_per_patient) if n else 0.0,
            'meals_total': int(sums['n_meals']),
            'meals_per_patient': sums['n_meals'] / n if n else 0.0,
            'meals_per_day': (sums['n_meals'] / n / days_per_patient) if n else 0.0,
            'insulin_total_u': total_insulin_u,
            'insulin_total_litres_u100': total_insulin_u / 1e5,
            'insulin_per_patient_u': total_insulin_u / n if n else 0.0,
            'insulin_per_day_u': (total_insulin_u / n / days_per_patient) if n else 0.0,
            'basal_total_u': total_basal_u,
            'bolus_total_u': total_bolus_u,
            'basal_frac': total_basal_u / total_insulin_u if total_insulin_u else 0.0,
            'bolus_frac': total_bolus_u / total_insulin_u if total_insulin_u else 0.0,
            'boluses_total': int(sums['n_boluses']),
            'boluses_per_day': (sums['n_boluses'] / n / days_per_patient) if n else 0.0,
            'exercise_active_frac': frac(sums['ex_active_steps']),
            'exercise_sessions_per_patient': sums['n_ex_sessions'] / n if n else 0.0,
        },
        'glycemia': {
            'bg_mean': acc.sums['sum_bg'] / total_steps if total_steps else 0.0,
            'bg_min': acc.bg_min,
            'bg_max': acc.bg_max,
            'frac_below54': frac(sums['below54']),
            'frac_below70': frac(sums['below70']),
            'frac_in_range': frac(sums['in_range']),
            'frac_above180': frac(sums['above180']),
            'frac_above250': frac(sums['above250']),
            # Temporal class ratios (partition of CGM time; sum to 1).
            'class_hypo': frac(sums['below70']),
            'class_in_range': frac(sums['in_range']),
            'class_hyper': frac(sums['above180']),
        },
        'distribution': _pool_distribution(bg_dist),
        'baseline': baseline,
        'icr': {
            'mean': acc.icr_sum / n if n else 0.0,
            'min': acc.icr_min if n else 0.0,
            'max': acc.icr_max if n else 0.0,
        },
        'sampling': {
            'total_simulations': acc.total_attempts,
            'accept_rate': n / acc.total_attempts if acc.total_attempts else 0.0,
            'mean_attempts_per_row': acc.total_attempts / n if n else 0.0,
            'n_fallback_rows': acc.n_fallback,
            'n_hypo_designated': acc.n_hypo_rows,
            'n_hypo_qualified': acc.n_hypo_qualified,
        },
        'storage': {
            'inventory': inventory,
            'raw_channel_bytes': raw_total,
            'compressed_channel_bytes': compressed_total,
            'icr_bytes': icr_bytes,
            'disk_total_bytes': disk_total,
            'compression_ratio': raw_total / max(compressed_total, 1),
        },
    }


def _fmt_bytes(n: float) -> str:
    for unit, scale in (('TB', 1e12), ('GB', 1e9), ('MB', 1e6), ('KB', 1e3)):
        if n >= scale:
            return f'{n / scale:.2f} {unit}'
    return f'{n:.0f} B'


def _fmt_mass_g(grams: float) -> str:
    """Grams -> a human unit (tonnes / kg / g) keeping the mantissa in [1, 1000)."""
    for unit, scale in (('tonnes', 1e6), ('kg', 1e3)):
        if grams >= scale:
            return f'{grams / scale:,.1f} {unit}'
    return f'{grams:,.0f} g'


def _fmt_vol_l(litres: float) -> str:
    """Litres -> L or mL, adaptively."""
    if litres >= 1.0:
        return f'{litres:,.0f} L' if litres >= 100 else f'{litres:,.1f} L'
    ml = litres * 1000
    if ml >= 10:
        return f'{ml:,.0f} mL'
    return f'{ml:,.1f} mL' if ml >= 0.1 else f'{ml * 1000:,.0f} µL'


def _fmt_years(years: float) -> str:
    if years >= 10:
        return f'{years:,.0f} years'
    if years >= 1:
        return f'{years:.1f} years'
    return f'{years:.2f} years'


def _render_comparison(report: dict[str, Any]) -> list[str]:
    """
    Render the 'Distribution vs the baseline simulator' section.

    Compares the pooled distribution of this cache against the unbiased simulator
    baseline loaded from the diff report (``datasets.Sim`` in stats.json). Empty
    if no baseline was available.
    """
    dist = report.get('distribution') or {}
    base = report.get('baseline')
    gly = report['glycemia']
    if not base or not dist:
        return []

    # (label, pool value, baseline value, format, delta-as-points?)
    rows = [
        ('Mean BG (mg/dL)', dist['mean'], base['mean'], '.1f', False),
        ('Median', dist['median'], base['median'], '.1f', False),
        ('Std', dist['std'], base['std'], '.1f', False),
        ('CV (%)', dist['cv_pct'], base['cv_pct'], '.1f', False),
        ('Skew', dist['skew'], base['skew'], '.2f', False),
        ('p5', dist['p5'], base['p5'], '.1f', False),
        ('p25', dist['p25'], base['p25'], '.1f', False),
        ('p75', dist['p75'], base['p75'], '.1f', False),
        ('p95', dist['p95'], base['p95'], '.1f', False),
        ('Time <54 (%)', 100 * gly['frac_below54'], 100 * base['frac_below54'], '.2f', True),
        ('Time <70 (%)', 100 * gly['frac_below70'], 100 * base['frac_below70'], '.2f', True),
        ('TIR 70–180 (%)', 100 * gly['frac_in_range'], 100 * base['frac_in_range'], '.2f', True),
        ('Time >180 (%)', 100 * gly['frac_above180'], 100 * base['frac_above180'], '.2f', True),
        ('Time >250 (%)', 100 * gly['frac_above250'], 100 * base['frac_above250'], '.2f', True),
        ('LBGI (hypo risk)', dist['lbgi'], base['lbgi'], '.2f', False),
        ('HBGI (hyper risk)', dist['hbgi'], base['hbgi'], '.2f', False),
    ]

    L: list[str] = []
    L.append('## Distribution vs the baseline simulator')
    L.append('')
    L.append('| Metric | This pool | Baseline sim | Δ |')
    L.append('|---|---|---|---|')
    for label, pv, bv, fmt, _ in rows:
        delta = pv - bv
        L.append(f'| {label} | {pv:{fmt}} | {bv:{fmt}} | {delta:+{fmt}} |')
    L.append('')
    return L


def _render_dataset_md(report: dict[str, Any]) -> str:
    """Render DATASET.md from the finalized report (neutral, observational)."""
    c = report['corpus']
    cons = report['consumption']
    gly = report['glycemia']
    icr = report['icr']
    smp = report['sampling']
    st = report['storage']
    p = report['params']

    def pct(x: float) -> str:
        return f'{100.0 * x:.2f}%'

    lines: list[str] = []
    lines.append('# Pregenerated Dataset')
    lines.append('')
    lines.append('| Quantity | Value | Basis |')
    lines.append('|---|---|---|')
    lines.append(f'| **Patients (trajectories)** | **{c["n_patients"]:,}** | exact (`meta.json`) |')
    lines.append(f'| Timesteps / patient | {c["n_timesteps"]} @ {int(c["dt_minutes"])} min | exact |')
    lines.append(f'| Duration / patient | {c["keep_hours"]:.1f} h ≈ {c["days_per_patient"]:.2f} days | exact |')
    lines.append(f'| **Total CGM readings** | **{c["total_readings"]:,}** | exact ({c["n_patients"]:,} × {c["n_timesteps"]}) |')
    lines.append(f'| **Total CGM hours** | **{c["total_cgm_hours"]:,.0f} h** | exact |')
    lines.append(f'| **Total CGM years** | **≈ {_fmt_years(c["total_cgm_years"])}** | exact (= patient-years) |')
    lines.append(f'| **Disk footprint** | **{_fmt_bytes(st["disk_total_bytes"])}** | exact (compressed `.b2nd` + `icr.npy`) |')
    lines.append(f'| Uncompressed channels | {_fmt_bytes(st["raw_channel_bytes"])} | exact ({st["compression_ratio"]:.2f}× ratio) |')
    lines.append(f'| **Carbs appeared** | **{_fmt_mass_g(cons["carb_total_g"])}** | integral of carb-appearance, incl. hypo rescue ({cons["carb_per_patient_g"]:.0f} g/patient, {cons["carb_per_day_g"]:.1f} g/day) |')
    lines.append(f'| **Carb-appearance events** | **{cons["meals_total"]:,}** | upward crossings, incl. hypo rescue ({cons["meals_per_patient"]:.2f}/patient, {cons["meals_per_day"]:.2f}/day) |')
    lines.append(f'| **Insulin delivered** | **{cons["insulin_total_u"]:,.0f} U** (≈ {_fmt_vol_l(cons["insulin_total_litres_u100"])} of U-100) | integral of delivery ({cons["insulin_per_patient_u"]:.1f} U/patient, {cons["insulin_per_day_u"]:.1f} U/day) |')
    lines.append(f'| └ basal : bolus split | {pct(cons["basal_frac"])} : {pct(cons["bolus_frac"])} | exact (separate channels summed) |')
    lines.append(f'| Boluses | {cons["boluses_total"]:,} ({cons["boluses_per_day"]:.2f}/day) | threshold-crossings (merges overlapping PK tails) |')
    lines.append(f'| Exercise | active {pct(cons["exercise_active_frac"])} of time; {cons["exercise_sessions_per_patient"]:.2f} sessions/patient | exact |')
    lines.append(f'| BG mean | {gly["bg_mean"]:.1f} mg/dL | exact |')
    lines.append(f'| Time-in-range 70–180 | {pct(gly["frac_in_range"])} | exact |')
    lines.append(f'| Time <70 / <54 (hypo) | {pct(gly["frac_below70"])} / {pct(gly["frac_below54"])} | exact |')
    lines.append(f'| Time >180 / >250 (hyper) | {pct(gly["frac_above180"])} / {pct(gly["frac_above250"])} | exact |')
    lines.append(f'| ICR (carb ratio) | {icr["mean"]:.1f} g/U mean (≈ {icr["min"]:.1f}–{icr["max"]:.1f}) | exact |')
    lines.append('')

    lines.append('## Temporal class ratios')
    lines.append('')
    lines.append('| Class | Band (mg/dL) | Share of time |')
    lines.append('|---|---|---|')
    lines.append(f'| Hypo | < 70 | {pct(gly["class_hypo"])} |')
    lines.append(f'| In-range | 70–180 | {pct(gly["class_in_range"])} |')
    lines.append(f'| Hyper | > 180 | {pct(gly["class_hyper"])} |')
    lines.append('')

    lines.extend(_render_comparison(report))

    lines.append('## Per-channel inventory')
    lines.append('')
    lines.append('| Channel | dtype | Uncompressed | Compressed | Ratio |')
    lines.append('|---|---|---|---|---|')
    for name in CHANNEL_NAMES:
        inv = st['inventory'][name]
        r = inv['raw_bytes'] / max(inv['compressed_bytes'], 1)
        lines.append(
            f'| `{name}` | {np.dtype(CHANNEL_DTYPES[name]).name} | '
            f'{_fmt_bytes(inv["raw_bytes"])} | {_fmt_bytes(inv["compressed_bytes"])} | {r:.2f}× |'
        )
    lines.append(f'| `icr` | float32 | {_fmt_bytes(st["icr_bytes"])} | {_fmt_bytes(st["icr_bytes"])} | 1.00× (raw) |')
    lines.append('')

    lines.append('## Generation parameters')
    lines.append('')
    lines.append('| Parameter | Value |')
    lines.append('|---|---|')
    lines.append(f'| Post-warmup window | {p["sim_hours"]:.1f} h ({c["n_timesteps"]} steps) |')
    lines.append(f'| Warmup discarded | {p["warmup_hours"]:.1f} h |')
    lines.append(f'| Seed salt | {p["seed_salt"]} |')
    lines.append(f'| Rail filter | discard bg_observed ≥ {p["rail_high"]:.0f} or ≤ {p["rail_low"]:.0f} mg/dL |')
    lines.append(f'| Hypo oversampling | prob {p["hypo_oversample"]:.3g}, ≥ {pct(p["hypo_min_frac"])} of time < {p["hypo_threshold"]:.0f} mg/dL |')
    lines.append(f'| Rejection cap | {p["max_attempts"]} attempts/row (soft) |')
    lines.append(f'| Compression | blosc2 zstd-{p["zstd_clevel"]} + byte-shuffle, {p["rows_per_chunk"]} rows/chunk |')
    lines.append('')
    lines.append('### Sampling')
    lines.append('')
    lines.append('| Quantity | Value |')
    lines.append('|---|---|')
    lines.append(f'| Trajectories simulated | {smp["total_simulations"]:,} for {c["n_patients"]:,} kept rows |')
    lines.append(f'| Accept rate | {pct(smp["accept_rate"])} (mean {smp["mean_attempts_per_row"]:.2f} attempts/row) |')
    lines.append(f'| Hypo-designated rows | {smp["n_hypo_designated"]:,} ({smp["n_hypo_qualified"]:,} met the target within the soft cap) |')
    lines.append(f'| Hypo rows below target (soft-cap fallback) | {smp["n_fallback_rows"]:,} |')
    lines.append('')
    return '\n'.join(lines)


# ============================================================================
# BUILD
# ============================================================================

def build_cache(
    out_dir: str,
    pool_size: int,
    sim_hours: float,
    warmup_hours: float,
    n_jobs: int,
    rows_per_chunk: int = DEFAULT_ROWS_PER_CHUNK,
    zstd_clevel: int = DEFAULT_ZSTD_CLEVEL,
    hypo_oversample: float = 0.0,
    hypo_min_frac: float = DEFAULT_HYPO_MIN_FRAC,
    hypo_threshold: float = BG_LOW,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    rail_high: float = DEFAULT_RAIL_HIGH,
    rail_low: float = DEFAULT_RAIL_LOW,
    seed_salt: int = 0,
    dataset_md: str = 'DATASET.md',
    baseline_stats: str | None = DEFAULT_BASELINE_STATS,
    force: bool = False,
) -> dict[str, Any]:
    """Build the cache at ``out_dir`` atomically and emit ``dataset_md``."""
    if pool_size < 1:
        raise ValueError(f'pool_size must be >= 1, got {pool_size}')
    if not rail_low < rail_high:
        raise ValueError(f'rail_low ({rail_low}) must be < rail_high ({rail_high})')
    if not 0.0 <= hypo_oversample <= 1.0:
        raise ValueError(f'hypo_oversample must be in [0, 1], got {hypo_oversample}')
    if not 0.0 <= hypo_min_frac <= 1.0:
        raise ValueError(f'hypo_min_frac must be in [0, 1], got {hypo_min_frac}')
    if max_attempts < 1:
        raise ValueError(f'max_attempts must be >= 1, got {max_attempts}')
    if os.path.exists(out_dir):
        if not force:
            raise FileExistsError(
                f'Cache directory {out_dir!r} already exists. Pass --force to overwrite.'
            )
        shutil.rmtree(out_dir)

    partial_dir = out_dir + '.partial'
    if os.path.exists(partial_dir):
        shutil.rmtree(partial_dir)
    os.makedirs(partial_dir)

    warmup_steps = _steps_for_hours(warmup_hours)
    keep_steps = _steps_for_hours(sim_hours)
    if keep_steps < 1:
        raise ValueError(f'sim_hours={sim_hours} yields {keep_steps} steps; need >= 1')
    effective_rpc = _resolve_rows_per_chunk(rows_per_chunk, pool_size, keep_steps)

    cfg = RowConfig(
        warmup_steps=warmup_steps,
        keep_steps=keep_steps,
        max_attempts=max_attempts,
        rail_high=rail_high,
        rail_low=rail_low,
        hypo_prob=hypo_oversample,
        hypo_min_frac=hypo_min_frac,
        hypo_threshold=hypo_threshold,
        seed_salt=seed_salt,
        event_refractory_steps=max(1, int(round(EVENT_REFRACTORY_MIN / DT_MINUTES))),
    )

    print(f'Pool size:          {pool_size:,}')
    print(f'Post-warmup window: {sim_hours} h ({keep_steps} steps/patient)')
    print(f'Warmup discarded:   {warmup_hours} h ({warmup_steps} steps)')
    print(f'Rail filter:        discard bg_observed >= {rail_high} or <= {rail_low} mg/dL')
    print(f'Hypo oversampling:  prob {hypo_oversample} (>= {hypo_min_frac} of time < {hypo_threshold:g} mg/dL)')
    print(f'Rejection cap:      {max_attempts} attempts/row (soft; hypo target)')
    print(f'Workers:            {n_jobs}')
    print(f'Compression:        blosc2 zstd-{zstd_clevel} + byte-shuffle, {effective_rpc} rows/chunk')
    print(f'Output dir:         {out_dir} (building in {partial_dir})')
    print()

    acc = StatAccumulator()
    icr_arr = np.zeros(pool_size, dtype=np.float32)
    t0 = time.time()

    try:
        # Create the staging memmaps (writes .npy headers, sparse-allocates the
        # body), then drop the parent handles before forking workers so the huge
        # mappings are not inherited -- each worker opens its own r+ handle.
        for name in CHANNEL_NAMES:
            np.lib.format.open_memmap(
                os.path.join(partial_dir, f'{name}.npy'),
                mode='w+', dtype=CHANNEL_DTYPES[name], shape=(pool_size, keep_steps),
            )
        gc.collect()

        completed = 0

        def _store(stats: dict[str, Any]) -> None:
            nonlocal completed
            acc.add(stats)
            icr_arr[stats['idx']] = stats['icr']
            completed += 1
            if completed == 1 or completed % 100 == 0 or completed == pool_size:
                elapsed = time.time() - t0
                rate = completed / max(elapsed, 1e-9)
                eta = (pool_size - completed) / max(rate, 1e-9)
                print(
                    f'  [{completed:>7,}/{pool_size:,}] {rate:7.1f} patients/s, '
                    f'elapsed {elapsed:6.1f}s, ETA {eta:6.1f}s',
                    flush=True,
                )

        if n_jobs <= 1:
            _worker_init(partial_dir, cfg)
            for i in range(pool_size):
                _store(_worker(i))
        else:
            ctx = mp.get_context('fork' if 'fork' in mp.get_all_start_methods() else 'spawn')
            chunksize = max(1, min(16, pool_size // (n_jobs * 4) or 1))
            with ctx.Pool(
                processes=n_jobs, initializer=_worker_init, initargs=(partial_dir, cfg),
            ) as pool:
                for stats in pool.imap_unordered(_worker, range(pool_size), chunksize=chunksize):
                    _store(stats)

        gen_secs = time.time() - t0
        print()
        print(f'All {pool_size:,} rows generated in {gen_secs:.1f}s '
              f'({pool_size / max(gen_secs, 1e-9):.1f} patients/s).')

        # Drop any staging memmap handles the single-threaded path left open in
        # this process (the MP path's handles die with the workers). A live mmap
        # pins the inode after os.remove, so peak disk would otherwise be
        # raw + compressed instead of ~max(raw, compressed).
        _WORKER.clear()
        gc.collect()

        print(f'Transcoding {len(CHANNEL_NAMES)} channels to blosc2 (rows/chunk={effective_rpc})...', flush=True)

        t_trans = time.time()
        inventory, bg_dist, norm_acc = _transcode_to_blosc2(
            partial_dir, pool_size, keep_steps, effective_rpc, zstd_clevel, n_jobs
        )
        print(f'Transcode finished in {time.time() - t_trans:.1f}s.')

        baseline = _load_baseline_stats(baseline_stats) if baseline_stats else None
        if baseline_stats and baseline is None:
            print(f'  (baseline comparison skipped: {baseline_stats} unavailable)', flush=True)

        icr_path = os.path.join(partial_dir, 'icr.npy')
        np.save(icr_path, icr_arr)
        _fsync_file(icr_path)
        icr_bytes = os.path.getsize(icr_path)

        params = {
            'out_dir': out_dir,
            'pool_size': int(pool_size),
            'n_timesteps': int(keep_steps),
            'sim_hours': float(sim_hours),
            'warmup_hours': float(warmup_hours),
            'dt_minutes': float(DT_MINUTES),
            'hypo_oversample': float(hypo_oversample),
            'hypo_min_frac': float(hypo_min_frac),
            'hypo_threshold': float(hypo_threshold),
            'max_attempts': int(max_attempts),
            'rail_high': float(rail_high),
            'rail_low': float(rail_low),
            'seed_salt': int(seed_salt),
            'rows_per_chunk': int(effective_rpc),
            'zstd_clevel': int(zstd_clevel),
            'channels': list(CHANNEL_NAMES),
            'cache_format': CACHE_FORMAT_VERSION,
            'baseline_stats': baseline['source'] if baseline else None,
        }
        report = _finalize_report(acc, params, inventory, icr_bytes, bg_dist, baseline)
        # normalization_stats.json (the 3-channel {mean, std} the T1DMAI model
        # consumes) is written BEFORE the meta.json sentinel and fsync'd inside
        # the same atomic flow, so the sentinel never surfaces a cache missing
        # valid stats.
        norm_stats = _finalize_norm_stats(norm_acc)
        norm_stats_path = os.path.join(partial_dir, NORM_STATS_FILE)
        with open(norm_stats_path, 'w') as f:
            json.dump(norm_stats, f, indent=2)
        _fsync_file(norm_stats_path)
        # meta.json is the last file written -- its presence marks a complete
        # cache. fsync every file and the staging dir before the atomic rename so
        # a power loss cannot surface a sentinel-present, data-truncated cache.
        meta_path = os.path.join(partial_dir, 'meta.json')
        with open(meta_path, 'w') as f:
            json.dump(report, f, indent=2)
        _fsync_file(meta_path)
        _fsync_dir(partial_dir)

        os.rename(partial_dir, out_dir)
        _fsync_dir(os.path.dirname(os.path.abspath(out_dir)))
    except BaseException:
        gc.collect()
        if os.path.exists(partial_dir):
            shutil.rmtree(partial_dir, ignore_errors=True)
        raise

    if dataset_md:
        with open(dataset_md, 'w') as f:
            f.write(_render_dataset_md(report))
        print(f'Wrote dataset report to {dataset_md}')

    print(f'Wrote cache to {out_dir}')
    return report


def _print_class_ratios(report: dict[str, Any]) -> None:
    gly = report['glycemia']
    print()
    print('Temporal class ratios (fraction of CGM time):')
    print(f'  hypo     (< 70 mg/dL): {100 * gly["class_hypo"]:6.2f}%')
    print(f'  in-range (70-180)    : {100 * gly["class_in_range"]:6.2f}%')
    print(f'  hyper    (> 180 mg/dL): {100 * gly["class_hyper"]:6.2f}%')


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Pre-generate a compressed pool of T1DMSIM trajectories + a DATASET.md report.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--out-dir', type=str, default='simulator_cache',
                        help='Output directory for the cached arrays.')
    parser.add_argument('--pool-size', type=int, default=50000,
                        help='Number of patient trajectories to generate.')
    parser.add_argument('--sim-hours', type=float, default=DEFAULT_SIM_HOURS,
                        help='Post-warmup hours kept per patient.')
    parser.add_argument('--warmup-hours', type=float, default=DEFAULT_WARMUP_HOURS,
                        help='Hours discarded from the start of every run.')
    parser.add_argument('--jobs', type=int, default=max(1, (os.cpu_count() or 2) - 1),
                        help='Worker processes (1 disables multiprocessing).')
    parser.add_argument('--rows-per-chunk', type=int, default=DEFAULT_ROWS_PER_CHUNK,
                        help='Compressed-chunk granularity in rows. Larger = faster '
                             'parallel transcode + slightly better ratio, but more '
                             'bytes decompressed per random-row read.')
    parser.add_argument('--zstd-clevel', type=int, default=DEFAULT_ZSTD_CLEVEL,
                        help='zstd compression level (1-9).')
    parser.add_argument('--hypo-oversample', type=float, default=0.0,
                        help='Extent of hypo oversampling: fraction of rows forced to be '
                             'hypo-rich via seed rejection sampling (0 disables).')
    parser.add_argument('--hypo-min-frac', type=float, default=DEFAULT_HYPO_MIN_FRAC,
                        help='Extent per row: min fraction of time below --hypo-threshold '
                             'a hypo-designated row must reach.')
    parser.add_argument('--hypo-threshold', type=float, default=BG_LOW,
                        help='BG cutoff (mg/dL) defining "hypo" for oversampling; lower it '
                             '(e.g. 54) to oversample more severe hypoglycemia.')
    parser.add_argument('--max-attempts', type=int, default=DEFAULT_MAX_ATTEMPTS,
                        help='Soft rejection cap per row for meeting the hypo target.')
    parser.add_argument('--rail-high', type=float, default=DEFAULT_RAIL_HIGH,
                        help='Discard a window with any bg_observed >= this (mg/dL).')
    parser.add_argument('--rail-low', type=float, default=DEFAULT_RAIL_LOW,
                        help='Discard a window with any bg_observed <= this (mg/dL).')
    parser.add_argument('--seed-salt', type=int, default=0,
                        help='Salt folded into every seed; change to draw an independent corpus.')
    parser.add_argument('--dataset-md', type=str, default='DATASET.md',
                        help='Path for the rendered statistics report (empty to skip).')
    parser.add_argument('--baseline-stats', type=str, default=DEFAULT_BASELINE_STATS,
                        help='diff/stats.json to source the unbiased-simulator baseline for '
                             'the DATASET.md distribution comparison.')
    parser.add_argument('--no-baseline', action='store_true',
                        help='Skip the baseline distribution comparison in DATASET.md.')
    parser.add_argument('--force', action='store_true',
                        help='Delete an existing --out-dir before building.')
    parser.add_argument('--rerender-report', type=str, default=None, metavar='CACHE_DIR',
                        help='Re-emit DATASET.md for an already-built cache from its '
                             'meta.json, without regenerating any trajectories.')
    args = parser.parse_args()

    if args.rerender_report:
        meta_path = os.path.join(args.rerender_report, 'meta.json')
        with open(meta_path) as f:
            report = json.load(f)
        md_path = os.path.join(args.rerender_report, args.dataset_md)
        with open(md_path, 'w') as f:
            f.write(_render_dataset_md(report))
        print(f'Wrote dataset report to {md_path}')
        _print_class_ratios(report)
        return 0

    report = build_cache(
        out_dir=args.out_dir,
        pool_size=args.pool_size,
        sim_hours=args.sim_hours,
        warmup_hours=args.warmup_hours,
        n_jobs=args.jobs,
        rows_per_chunk=args.rows_per_chunk,
        zstd_clevel=args.zstd_clevel,
        hypo_oversample=args.hypo_oversample,
        hypo_min_frac=args.hypo_min_frac,
        hypo_threshold=args.hypo_threshold,
        max_attempts=args.max_attempts,
        rail_high=args.rail_high,
        rail_low=args.rail_low,
        seed_salt=args.seed_salt,
        dataset_md=args.dataset_md,
        baseline_stats=None if args.no_baseline else args.baseline_stats,
        force=args.force,
    )
    _print_class_ratios(report)
    return 0


if __name__ == '__main__':
    sys.exit(main())
