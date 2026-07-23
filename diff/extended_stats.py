"""Extended statistical metrics for the T1DMSIM-vs-real comparison.

Self-contained, pure functions layered on top of the core metrics in
build_report.py. Four families, all added to give the sim-vs-real comparison
more resolution and to make its strengths/weaknesses objective:

  1. Extra two-sample distances (energy, Cramer-von Mises, Anderson-Darling,
     total variation, Hellinger, histogram overlap) so no single distance
     drives the verdict.
  2. Cadence-fair recompute: cadence-sensitive metrics (MAGE, delta-SD, CONGA,
     short-lag ACF, episodes) evaluated for EVERY cohort on one common 15-min
     grid, so the 15-min Shanghai cohort is finally apples-to-apples with the
     5-min cohorts and the simulator.
  3. Temporal structure: spectral entropy (Welch PSD), Poincare SD1/SD2, DFA
     scaling exponent (Hurst), ACF e-folding time, and a glycemic-band Markov
     transition matrix with per-band mean dwell time.
  4. Cross-seed bootstrap confidence intervals (operationalising "many seeds to
     minimise noise") and a standardised per-metric gap score
     z = (sim - mean_real) / sd_between_real_cohorts that ranks where the
     simulator sits inside vs outside the real cohorts' own spread.

Every function ignores NaN, treats a bridged sensor gap as a true
discontinuity (never joining samples across it), and is deterministic given a
fixed rng_seed. mg/dL throughout.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as sps
from scipy import signal as spsig

# Glycemic bands, low->high. Edges match time_in_ranges(): [<54, 54-70, 70-180,
# 180-250, >250]. A value lands in the first band whose upper edge it is below.
BAND_EDGES = (54.0, 70.0, 180.0, 250.0)
BAND_LABELS = ("TBR2", "TBR1", "TIR", "TAR1", "TAR2")

# Slack on each endpoint of the real cohorts' [min,max] envelope, as a fraction
# of the envelope's own span. The endpoints are order statistics of three
# cohorts, not exact bounds, so a sim value overshooting one of them by a
# thousandth of the inter-cohort spread is a tie, not an exceedance.
ENVELOPE_REL_TOL = 0.01


def _clean(x):
    x = np.asarray(x, dtype=float)
    return x[~np.isnan(x)]


def to_common_grid(bg, step_min, target_min=15):
    """Decimate a series to a `target_min`-minute effective interval.

    5-min cohorts (Ohio/AZT1D/Sim) get stride 3 -> 15 min; a natively-15-min
    Shanghai record (step_min=15) is returned unchanged. NaN gap cells are
    preserved on the decimated grid so downstream code can still see the
    discontinuities. Never up-samples (if step_min >= target_min, pass through).
    """
    bg = np.asarray(bg, dtype=float)
    stride = max(1, int(round(target_min / step_min)))
    return bg[::stride], step_min * stride


def _longest_contiguous(bg):
    """Longest run of consecutive non-NaN samples (a 1-D float array)."""
    bg = np.asarray(bg, dtype=float)
    valid = ~np.isnan(bg)
    best_lo = best_hi = 0
    i = 0
    n = len(valid)
    while i < n:
        if valid[i]:
            j = i
            while j < n and valid[j]:
                j += 1
            if j - i > best_hi - best_lo:
                best_lo, best_hi = i, j
            i = j
        else:
            i += 1
    return bg[best_lo:best_hi]


# ============================================================================
# Family 1 — extra two-sample distances
# ============================================================================
def _subsample(x, max_n, rng):
    if len(x) <= max_n:
        return x
    idx = rng.choice(len(x), size=max_n, replace=False)
    idx.sort()
    return x[idx]


def extra_distances(a, b, bins=None, common_n=None, max_n=60000, rng_seed=0):
    """Distances between two pooled BG vectors, complementing KS/Wasserstein/JS.

    - energy_distance: sqrt of the energy statistic (a proper metric on
      distributions), in mg/dL-like units. Computed on the FULL vectors (it is
      sample-size-stable).
    - cramer_von_mises: the 2-sample CvM omega^2 statistic (whole-CDF L2, more
      tail-sensitive than KS which is a sup norm).
    - anderson_darling: the k-sample AD statistic (extra weight in the tails).
    - total_variation / hellinger / overlap: from 5-mg/dL histograms over the
      full support so no tail is dropped (TV in [0,1], Hellinger in [0,1],
      overlap = 1 - TV). Full vectors, sample-size-stable.

    The rank-based CvM/AD STATISTICS grow ~linearly with n under H1, so to be
    comparable ACROSS cohort pairs they must be evaluated at one common n. Pass
    `common_n` (the smallest pooled cohort size, capped at `max_n`) so EVERY
    pair subsamples both arms to exactly that n; otherwise pairs with a smaller
    arm would read a spuriously smaller statistic. KS/Wasserstein (computed
    elsewhere) stay on the full vectors.
    """
    a = _clean(a)
    b = _clean(b)
    out = {"energy": float("nan"), "cramer_von_mises": float("nan"),
           "anderson_darling": float("nan"), "total_variation": float("nan"),
           "hellinger": float("nan"), "overlap": float("nan"),
           "subsample_n": 0}
    if len(a) == 0 or len(b) == 0:
        return out
    rng = np.random.default_rng(rng_seed)
    target = max_n if common_n is None else common_n
    target = int(min(target, len(a), len(b)))
    aa = _subsample(a, target, rng)
    bb = _subsample(b, target, rng)
    out["subsample_n"] = int(target)
    out["energy"] = float(sps.energy_distance(a, b))
    try:
        out["cramer_von_mises"] = float(sps.cramervonmises_2samp(aa, bb).statistic)
    except Exception:
        pass
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out["anderson_darling"] = float(sps.anderson_ksamp([aa, bb]).statistic)
    except Exception:
        pass
    if bins is None:
        bins = np.arange(0, 601, 5)
    pa, _ = np.histogram(a, bins=bins, density=False)
    pb, _ = np.histogram(b, bins=bins, density=False)
    pa = pa / pa.sum() if pa.sum() else pa.astype(float)
    pb = pb / pb.sum() if pb.sum() else pb.astype(float)
    out["total_variation"] = float(0.5 * np.sum(np.abs(pa - pb)))
    out["hellinger"] = float(np.sqrt(0.5 * np.sum((np.sqrt(pa) - np.sqrt(pb)) ** 2)))
    out["overlap"] = float(np.sum(np.minimum(pa, pb)))
    return out


# ============================================================================
# Family 2 — cadence-fair recompute (common 15-min grid)
# ============================================================================
def cadence_fair(bg, step_min, mage_fn, conga_fn, acf_fn, episode_fn,
                 target_min=15):
    """Recompute cadence-sensitive metrics on a common `target_min` grid.

    `bg` is a record's regularised series at its native `step_min` (may hold
    NaN gaps). It is decimated to `target_min` (Shanghai already there) and the
    supplied primitives — the SAME audited implementations used elsewhere — are
    run on the decimated series so Shanghai (15 min) and the 5-min cohorts are
    finally compared at one cadence.

    Returns per-record scalars: MAGE, Delta-SD (one-step at target cadence),
    CONGA-1h, ACF at 30/60/120 min, and hypo/hyper episodes per observed day.
    """
    grid, eff = to_common_grid(bg, step_min, target_min)
    n_obs = int(np.sum(~np.isnan(grid)))
    days = (n_obs * eff) / (60.0 * 24.0)
    d = np.diff(grid)
    d = d[~np.isnan(d)]
    acf = acf_fn(grid, eff, [30, 60, 120])
    h = episode_fn(grid, 70, True, eff)
    H = episode_fn(grid, 180, False, eff)
    return {
        "eff_step_min": eff,
        "mage": mage_fn(grid, step_min=eff),
        "delta_std": float(np.std(d)) if len(d) else float("nan"),
        "conga_1h": conga_fn(grid, 1, eff),
        "acf_30m": acf.get(30, float("nan")),
        "acf_60m": acf.get(60, float("nan")),
        "acf_120m": acf.get(120, float("nan")),
        "hypo_per_day": (len(h) / days) if days else float("nan"),
        "hyper_per_day": (len(H) / days) if days else float("nan"),
    }


# ============================================================================
# Family 3 — temporal structure
# ============================================================================
def poincare_sd(bg, step_min):
    """Poincare (lag-1 return-map) descriptors.

    SD1 = short-term variability = sqrt(0.5 * Var(dBG)); SD2 = long-term
    variability = sqrt(2*Var(BG) - 0.5*Var(dBG)); ratio SD1/SD2. dBG is taken
    within contiguous segments (diff, then drop gap-touching pairs) so a bridged
    gap never contributes a spurious jump. Cadence-dependent, so callers should
    pass a common-grid series for cross-cohort comparison.
    """
    bg = np.asarray(bg, dtype=float)
    d = np.diff(bg)
    d = d[~np.isnan(d)]
    x = _clean(bg)
    if len(d) < 2 or len(x) < 2:
        return {"sd1": float("nan"), "sd2": float("nan"), "sd_ratio": float("nan")}
    var_d = float(np.var(d))
    var_x = float(np.var(x))
    sd1 = float(np.sqrt(0.5 * var_d))
    sd2_sq = 2.0 * var_x - 0.5 * var_d
    sd2 = float(np.sqrt(sd2_sq)) if sd2_sq > 0 else float("nan")
    ratio = float(sd1 / sd2) if sd2 and np.isfinite(sd2) and sd2 > 0 else float("nan")
    return {"sd1": sd1, "sd2": sd2, "sd_ratio": ratio}


def spectral_entropy(bg, step_min, nperseg=256):
    """Normalised spectral entropy of the Welch PSD (0 = single tone, 1 = white).

    Computed on the longest contiguous segment (Welch needs a gap-free window).
    The PSD is normalised to a probability distribution over frequency and the
    Shannon entropy is divided by log(#bins) so the value is in [0,1] and
    comparable across records. Also returns the spectral centroid (mean
    frequency, cycles/hour) as a coarse "where is the power" summary.
    """
    seg = _longest_contiguous(bg)
    nan = {"spectral_entropy": float("nan"), "spectral_centroid_cph": float("nan")}
    if len(seg) < 32:
        return nan
    seg = seg - seg.mean()
    fs = 60.0 / step_min  # samples per hour
    nps = int(min(nperseg, len(seg)))
    f, pxx = spsig.welch(seg, fs=fs, nperseg=nps)
    pxx = pxx[1:]  # drop DC
    f = f[1:]
    tot = pxx.sum()
    if tot <= 0 or len(pxx) < 2:
        return nan
    p = pxx / tot
    ent = float(-np.sum(p * np.log(p + 1e-300)) / np.log(len(p)))
    centroid = float(np.sum(f * p))  # cycles per hour
    return {"spectral_entropy": ent, "spectral_centroid_cph": centroid}


def _dfa_one(seg, min_scale, max_scale, n_scales):
    """DFA scaling exponent on one gap-free window of length len(seg)."""
    n = len(seg)
    y = np.cumsum(seg - seg.mean())
    scales = np.unique(np.geomspace(min_scale, max_scale, n_scales).astype(int))
    scales = scales[scales >= min_scale]
    fs, ss = [], []
    for s in scales:
        nseg = n // s
        if nseg < 2:
            continue
        seg_rms = []
        t = np.arange(s)
        for k in range(nseg):
            w = y[k * s:(k + 1) * s]
            coef = np.polyfit(t, w, 1)
            resid = w - np.polyval(coef, t)
            seg_rms.append(np.mean(resid ** 2))
        fs.append(np.sqrt(np.mean(seg_rms)))
        ss.append(s)
    if len(ss) < 3:
        return float("nan")
    return float(np.polyfit(np.log(ss), np.log(fs), 1)[0])


def dfa_alpha(bg, step_min, chunk_len=256, min_scale=4, n_scales=10):
    """Detrended fluctuation analysis scaling exponent alpha (Hurst-like).

    alpha ~0.5 white noise, ~1.0 pink/1-f, ~1.5 brownian. Integrate the
    mean-removed series, split into non-overlapping windows of size s, linearly
    detrend each, take the RMS residual F(s), fit log F(s) vs log s.

    DFA's finite-size bias depends on record length, so the long gapless sim
    (one 6720-pt segment) and the short gappy real segments (~300-1300 pts)
    would read different alpha for identical dynamics purely from length. To
    remove that confound, alpha is measured on FIXED-LENGTH non-overlapping
    chunks (`chunk_len`) of the longest contiguous segment, over the fixed scale
    window [min_scale, chunk_len//4], and averaged. Every chunk — sim or real —
    uses the same n and the same scales, so the finite-size bias is identical
    and the alphas are directly comparable; the long sim simply averages over
    more chunks (lower variance, same mean). A record whose longest contiguous
    segment is shorter than one chunk returns NaN. chunk_len=256 ≈ 2.7 days on
    the 15-min grid, short enough to fit the shortest usable real segment.
    """
    seg = _longest_contiguous(bg)
    if len(seg) < chunk_len:
        return float("nan")
    max_scale = chunk_len // 4
    n_chunks = len(seg) // chunk_len
    alphas = []
    for k in range(n_chunks):
        a = _dfa_one(seg[k * chunk_len:(k + 1) * chunk_len],
                     min_scale, max_scale, n_scales)
        if np.isfinite(a):
            alphas.append(a)
    return float(np.mean(alphas)) if alphas else float("nan")


def acf_efold_min(acf_dict, target=1.0 / np.e):
    """Lag (minutes) at which the pooled ACF first falls below `target`.

    Linear interpolation between the bracketing lags of the pooled-ACF dict
    (keys may be int or str minutes). Returns np.inf if it never drops below.
    """
    items = []
    for k, v in acf_dict.items():
        try:
            lag = float(k)
        except (TypeError, ValueError):
            continue
        if v is not None and np.isfinite(v):
            items.append((lag, float(v)))
    items.sort()
    if not items:
        return float("inf")
    prev_lag, prev_v = items[0]
    if prev_v < target:
        return prev_lag
    for lag, v in items[1:]:
        if v < target:
            if prev_v == v:
                return lag
            frac = (prev_v - target) / (prev_v - v)
            return float(prev_lag + frac * (lag - prev_lag))
        prev_lag, prev_v = lag, v
    return float("inf")


def _band_of(v):
    # Match the canonical time_in_ranges() convention in build_report.py:
    # TBR2 <54, TBR1 [54,70), TIR [70,180], TAR1 (180,250], TAR2 >250. The upper
    # edges (180, 250) belong to the LOWER band, so a value of exactly 180 is TIR
    # and exactly 250 is TAR1 (a plain `v < edge` on every edge misfiled both).
    if v < BAND_EDGES[0]:       # <54
        return 0
    if v < BAND_EDGES[1]:       # 54-70
        return 1
    if v <= BAND_EDGES[2]:      # 70-180 (180 inclusive)
        return 2
    if v <= BAND_EDGES[3]:      # 180-250 (250 inclusive)
        return 3
    return 4


def band_transition(bg, step_min, coarse_min=15):
    """5x5 row-stochastic transition matrix over glycemic bands + dwell times.

    Bands: TBR2<54, TBR1 54-70, TIR 70-180, TAR1 180-250, TAR2>250. The series
    is decimated to `coarse_min` and only consecutive, both-valid, adjacent-in-
    time sample pairs contribute a transition (a gap breaks the chain). Dwell
    time per band = coarse_min / (1 - P[i,i]) (mean geometric sojourn), in
    minutes. Returns the matrix, per-band occupancy, and dwell dict.
    """
    grid, eff = to_common_grid(bg, step_min, coarse_min)
    counts = np.zeros((5, 5), dtype=float)
    for i in range(len(grid) - 1):
        a, b = grid[i], grid[i + 1]
        if np.isnan(a) or np.isnan(b):
            continue
        counts[_band_of(a), _band_of(b)] += 1.0
    row = counts.sum(axis=1, keepdims=True)
    mat = np.divide(counts, row, out=np.zeros_like(counts), where=row > 0)
    occ = (row[:, 0] / row.sum()) if row.sum() else np.zeros(5)
    dwell = {}
    for i, lab in enumerate(BAND_LABELS):
        pii = mat[i, i]
        dwell[lab] = float(eff / (1.0 - pii)) if (row[i, 0] > 0 and pii < 1.0) else float("nan")
    return {"matrix": mat.tolist(), "counts": counts.tolist(),
            "labels": list(BAND_LABELS),
            "occupancy": occ.tolist() if hasattr(occ, "tolist") else list(occ),
            "dwell_min": dwell, "eff_step_min": eff,
            "n_transitions": float(counts.sum())}


def pool_transition_counts(count_matrices, coarse_min=15):
    """Sum per-record 5x5 count matrices into one pooled, row-stochastic matrix.

    Pooling COUNTS (not averaging normalised matrices) weights each record by
    its transition volume, the statistically correct pooled estimator. Dwell
    time per band = coarse_min / (1 - P[i,i]).
    """
    total = np.zeros((5, 5), dtype=float)
    for c in count_matrices:
        total += np.asarray(c, dtype=float)
    row = total.sum(axis=1, keepdims=True)
    mat = np.divide(total, row, out=np.zeros_like(total), where=row > 0)
    occ = (row[:, 0] / row.sum()) if row.sum() else np.zeros(5)
    dwell = {}
    for i, lab in enumerate(BAND_LABELS):
        pii = mat[i, i]
        dwell[lab] = float(coarse_min / (1.0 - pii)) if (row[i, 0] > 0 and pii < 1.0) else float("nan")
    return {"matrix": mat.tolist(), "labels": list(BAND_LABELS),
            "occupancy": occ.tolist() if hasattr(occ, "tolist") else list(occ),
            "dwell_min": dwell, "n_transitions": float(total.sum())}


def transition_matrix_distance(mat_a, mat_b):
    """Frobenius distance between two row-stochastic transition matrices."""
    a = np.asarray(mat_a, dtype=float)
    b = np.asarray(mat_b, dtype=float)
    if a.shape != b.shape:
        return float("nan")
    return float(np.sqrt(np.sum((a - b) ** 2)))


# ============================================================================
# Family 4 — cross-seed bootstrap + standardised gap score
# ============================================================================
def bootstrap_pooled(record_arrays, stat_fn, n_boot=400, rng_seed=0,
                     ci=(2.5, 97.5)):
    """Bootstrap a pooled statistic by resampling whole RECORDS with replacement.

    `record_arrays` is a list of per-record 1-D arrays (NaN already dropped).
    Each bootstrap replicate resamples the records (not individual samples, so
    the CI reflects between-record/between-seed uncertainty — the noise the user
    wants many seeds to suppress), concatenates them, and applies stat_fn to the
    pooled vector. Returns point estimate + percentile CI + replicate SD.

    With many sim seeds the CI is tight; with a handful of real patients it is
    honestly wide — the two are directly comparable.
    """
    arrays = [np.asarray(a, dtype=float) for a in record_arrays if len(a) > 0]
    if not arrays:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "se": float("nan"), "n_records": 0}
    point = float(stat_fn(np.concatenate(arrays)))
    rng = np.random.default_rng(rng_seed)
    m = len(arrays)
    reps = np.empty(n_boot, dtype=float)
    for k in range(n_boot):
        idx = rng.integers(0, m, size=m)
        reps[k] = stat_fn(np.concatenate([arrays[j] for j in idx]))
    lo, hi = np.percentile(reps, ci)
    return {"point": point, "lo": float(lo), "hi": float(hi),
            "se": float(np.std(reps, ddof=1)) if n_boot > 1 else float("nan"),
            "n_records": m}


def gap_score(sim_val, real_vals):
    """Standardised gap of the sim from the real cohorts' own spread.

    z = (sim - mean(real_vals)) / sd(real_vals) with the sample SD (ddof=1)
    across the real cohorts. |z|<1 => the sim sits inside the band the real
    cohorts span among themselves (a strength); |z|>2 => it sits outside all of
    them (a weakness). With only three real cohorts the SD is coarse, so also
    returns whether the sim is within the [min,max] real envelope, which needs
    no distributional assumption. That envelope is tested with a margin of
    ENVELOPE_REL_TOL of its own span on each endpoint, so a sim value that ties
    an endpoint to within a fraction of a percent of the inter-cohort spread —
    a difference below both the report's printed precision and the sampling
    noise of either arm — still counts as inside.
    """
    reals = [v for v in real_vals if v is not None and np.isfinite(v)]
    if len(reals) < 2 or sim_val is None or not np.isfinite(sim_val):
        return {"z": float("nan"), "mean_real": float("nan"),
                "sd_real": float("nan"), "envelope_lo": float("nan"),
                "envelope_hi": float("nan"), "envelope_tol": float("nan"),
                "within_envelope": None}
    reals = np.asarray(reals, dtype=float)
    mean_r = float(np.mean(reals))
    sd_r = float(np.std(reals, ddof=1))
    if sd_r > 0:
        z = float((sim_val - mean_r) / sd_r)
    elif sim_val == mean_r:
        z = 0.0
    else:  # degenerate real SD but sim differs: definitively outside, signed
        z = float("inf") if sim_val > mean_r else float("-inf")
    lo_r, hi_r = float(reals.min()), float(reals.max())
    tol = ENVELOPE_REL_TOL * (hi_r - lo_r)
    return {"z": z, "mean_real": mean_r, "sd_real": sd_r,
            "envelope_lo": lo_r, "envelope_hi": hi_r, "envelope_tol": tol,
            "within_envelope": bool(lo_r - tol <= sim_val <= hi_r + tol)}
