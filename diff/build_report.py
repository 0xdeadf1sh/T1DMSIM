"""Comprehensive statistical comparison of T1DMSIM vs three real CGM corpora.

Compares the simulator output against OhioT1DM, ShanghaiT1DM, and AZT1D.

Produces:
  diff/stats.json        — all computed numbers
  diff/figures/*.png     — figure set referenced from README.md
  diff/README.md         — templated markdown report (regenerated from stats)

The analysis intentionally goes beyond scripts/compare_all_datasets.py and the
existing README comparison block:

  - full percentile table + central moments (skew, excess kurtosis)
  - Kolmogorov-Smirnov, Wasserstein-1, Jensen-Shannon distances between BG
    distributions
  - LBGI / HBGI (Kovatchev risk indices), J-index, M-value, MAGE, CONGA-1h,
    CONGA-4h, MODD
  - autocorrelation across lags 5 min → 24 h
  - Δ-BG distribution (rate of change) moments and percentiles
  - hour-of-day mean ± 1σ envelopes
  - episode-level metrics: count/day, full duration percentiles, time-to-recover
  - per-patient TIR/TBR scatter (heterogeneity inside each cohort)
  - AZT1D-only insulin / carb behaviour panel (basal-rate distribution, bolus
    type breakdown, correction-vs-meal split, carbs / boluses per day)
  - cohort summary tables in JSON keyed by metric

The four datasets are read with the loaders in scripts/compare_all_datasets.py;
the simulator is exercised with the same warm-up convention as that script.
"""
from __future__ import annotations

import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sps

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DIFF = os.path.join(REPO_ROOT, "diff")
FIGS = os.path.join(DIFF, "figures")
os.makedirs(FIGS, exist_ok=True)

sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from compare_all_datasets import (  # noqa: E402
    load_ohio_patients, load_shanghai_patients,
    load_azt1d_patients, load_azt1d_events,
    regularize_bg, regularize_bg_15min, OHIO_DIR,
)
from simulator import T1DMSimulator  # noqa: E402
import extended_stats as es  # noqa: E402


# ----- palette -----
# Four cohorts: three real (Ohio, Shanghai, AZT1D) plus the simulator.
COL = {"Ohio": "#1f77b4", "Shanghai": "#ff7f0e",
       "AZT1D": "#2ca02c", "Sim": "#d62728"}
ORDER = ["Ohio", "Shanghai", "AZT1D", "Sim"]
REAL_ORDER = ["Ohio", "Shanghai", "AZT1D"]


# ============================================================================
# Glycemic metrics
# ============================================================================
def kovatchev_risk(bg):
    """LBGI / HBGI per Kovatchev (1997). Operates on mg/dL, ignores NaN."""
    bg = bg[~np.isnan(bg)]
    if len(bg) == 0:
        return float("nan"), float("nan")
    f = 1.509 * (np.log(np.clip(bg, 1, None)) ** 1.084 - 5.381)
    rl = 10 * np.minimum(f, 0.0) ** 2  # left (hypo) risk
    rh = 10 * np.maximum(f, 0.0) ** 2  # right (hyper) risk
    return float(np.mean(rl)), float(np.mean(rh))


def j_index(bg):
    bg = bg[~np.isnan(bg)]
    if len(bg) == 0:
        return float("nan")
    return float(0.001 * (np.mean(bg) + np.std(bg)) ** 2)


def m_value(bg, ref=120.0):
    bg = bg[~np.isnan(bg)]
    if len(bg) == 0:
        return float("nan")
    return float(np.mean(np.abs(10 * np.log10(np.clip(bg, 1, None) / ref)) ** 3))


def mage(bg, step_min=5):
    """Mean amplitude of glycemic excursions. Counts peak-trough excursions
    whose amplitude exceeds 1 standard deviation."""
    bg = bg[~np.isnan(bg)]
    if len(bg) < 10:
        return float("nan")
    sd = np.std(bg)
    extrema = [(0, bg[0])]
    for i in range(1, len(bg) - 1):
        if (bg[i] > bg[i - 1] and bg[i] > bg[i + 1]) or \
           (bg[i] < bg[i - 1] and bg[i] < bg[i + 1]):
            extrema.append((i, bg[i]))
    extrema.append((len(bg) - 1, bg[-1]))
    amps = []
    for k in range(1, len(extrema)):
        amp = abs(extrema[k][1] - extrema[k - 1][1])
        if amp > sd:
            amps.append(amp)
    return float(np.mean(amps)) if amps else float("nan")


def conga(bg, hours, step_min):
    """Continuous overall net glycemic action — std of differences over `hours`."""
    bg = np.asarray(bg, dtype=float)
    lag = int(round(hours * 60 / step_min))
    if lag <= 0 or lag >= len(bg):
        return float("nan")
    d = bg[lag:] - bg[:-lag]
    d = d[~np.isnan(d)]
    return float(np.std(d)) if len(d) > 1 else float("nan")


def modd(times, bg, step_min):
    """Mean of daily differences. Aligns BG samples 24h apart."""
    bg = np.asarray(bg, dtype=float)
    lag = int(round(24 * 60 / step_min))
    if lag >= len(bg):
        return float("nan")
    d = np.abs(bg[lag:] - bg[:-lag])
    d = d[~np.isnan(d)]
    return float(np.mean(d)) if len(d) > 0 else float("nan")


def sample_entropy(x, m=2, r_frac=0.2, step_min=5, interval_min=15, max_pts=4000):
    """Sample entropy SampEn(m, r), r = r_frac * std(x), measured at a FIXED
    effective sampling interval (`interval_min`) so it is comparable across records
    of different length and native cadence.

    The previous version random-subsampled to a fixed COUNT, which made the effective
    spacing scale with record length: for one and the same process a 60-day trace
    read a far higher SampEn than a 14-day one (the sim's SampEn rose monotonically
    0.34 -> 1.11 as the run grew from 10 to 70 days). Striding on the grid to a fixed
    interval removes that confound; SampEn is then N-stable, so a generous `max_pts`
    cap only bounds cost. Stride BEFORE dropping gap NaNs so the true interval is
    preserved across dropouts.

    Template matches are accumulated WITHIN contiguous (gap-free) segments only.
    Simply dropping NaNs would concatenate the samples on either side of every
    dropout, so a template window could straddle a bridged gap — joining points
    hours-to-days apart as if `interval_min` apart. Those false adjacencies inject
    near-random jumps that inflate SampEn for the gappy real cohorts while leaving
    the gapless simulator untouched, spuriously collapsing the real-vs-sim
    complexity gap. Splitting at NaNs removes that artefact; a common `r` from the
    pooled valid samples keeps every segment on one scale."""
    x = np.asarray(x, dtype=float)
    stride = max(1, int(round(interval_min / step_min)))
    x = x[::stride]
    # Cap to the first max_pts VALID samples (preserving gap structure) so both
    # gapless (sim) and gappy (real) records use the same point budget and cost
    # — a gapless record must not silently run the full uncapped series.
    valid_mask = ~np.isnan(x)
    if int(valid_mask.sum()) > max_pts:
        cum = np.cumsum(valid_mask)
        cutoff = int(np.searchsorted(cum, max_pts, side="left")) + 1
        x = x[:cutoff]
    valid = x[~np.isnan(x)]
    if len(valid) < m + 2:
        return float("nan")
    r = r_frac * float(np.std(valid))
    if r == 0:
        return float("nan")

    # Contiguous non-NaN runs of the (capped) strided series.
    segments = []
    cur = []
    for v in x:
        if np.isnan(v):
            if len(cur) >= m + 1:
                segments.append(np.asarray(cur, dtype=float))
            cur = []
        else:
            cur.append(v)
    if len(cur) >= m + 1:
        segments.append(np.asarray(cur, dtype=float))
    if not segments:
        return float("nan")

    def phi(mm):
        # max-norm template matching within each segment, exclude self-match;
        # a window is never allowed to cross a former gap.
        cnt = 0
        for seg in segments:
            if len(seg) < mm + 1:
                continue
            templates = np.lib.stride_tricks.sliding_window_view(seg, mm)
            ntemp = len(templates)
            for i in range(ntemp):
                dists = np.max(np.abs(templates[i + 1:] - templates[i]), axis=1)
                cnt += int(np.sum(dists <= r))
        return cnt

    a = phi(m + 1)
    b = phi(m)
    if a == 0 or b == 0:
        return float("nan")
    return float(-np.log(a / b))


def autocorr_lags(bg, step_min, lags_min):
    """Pearson autocorrelation at the requested lag list (in minutes)."""
    bg = np.asarray(bg, dtype=float)
    out = {}
    for L_min in lags_min:
        lag = int(round(L_min / step_min))
        if lag <= 0 or lag >= len(bg):
            out[L_min] = float("nan")
            continue
        x = bg[:-lag]
        y = bg[lag:]
        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() < 50:
            out[L_min] = float("nan")
        else:
            xm, ym = x[mask], y[mask]
            sx, sy = np.std(xm), np.std(ym)
            if sx == 0 or sy == 0:
                out[L_min] = float("nan")
            else:
                out[L_min] = float(np.mean((xm - xm.mean()) * (ym - ym.mean())) / (sx * sy))
    return out


def time_in_ranges(bg):
    bg = bg[~np.isnan(bg)]
    n = len(bg)
    if n == 0:
        return {}
    return {
        "TBR2_pct": float(100 * np.sum(bg < 54) / n),
        "TBR1_pct": float(100 * np.sum((bg >= 54) & (bg < 70)) / n),
        "TIR_pct": float(100 * np.sum((bg >= 70) & (bg <= 180)) / n),
        "TAR1_pct": float(100 * np.sum((bg > 180) & (bg <= 250)) / n),
        "TAR2_pct": float(100 * np.sum(bg > 250) / n),
    }


def central_moments(bg):
    bg = bg[~np.isnan(bg)]
    if len(bg) == 0:
        return {}
    return {
        "n": int(len(bg)),
        "mean": float(np.mean(bg)),
        "median": float(np.median(bg)),
        "std": float(np.std(bg)),
        "iqr": float(np.percentile(bg, 75) - np.percentile(bg, 25)),
        "cv_pct": float(100 * np.std(bg) / np.mean(bg)),
        "skew": float(sps.skew(bg)),
        "excess_kurt": float(sps.kurtosis(bg)),
        "min": float(np.min(bg)),
        "max": float(np.max(bg)),
    }


def percentile_row(bg, pcts=(1, 5, 10, 25, 50, 75, 90, 95, 99)):
    bg = bg[~np.isnan(bg)]
    if len(bg) == 0:
        return {f"p{p}": float("nan") for p in pcts}
    return {f"p{p}": float(np.percentile(bg, p)) for p in pcts}


def episode_durations(bg, threshold, below=True, step_min=5, min_minutes=15):
    bg = bg.copy()
    if below:
        bg[np.isnan(bg)] = threshold + 1.0
        mask = bg < threshold
    else:
        bg[np.isnan(bg)] = threshold - 1.0
        mask = bg > threshold
    out = []
    i = 0
    n = len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            dur = (j - i) * step_min
            if dur >= min_minutes:
                out.append(dur)
            i = j
        else:
            i += 1
    return out


def episode_recovery_time(bg, low_thresh=70, normal_thresh=80, step_min=5,
                          min_minutes=15):
    """For each hypo episode, time from first sample <70 to first subsequent
    sample ≥80. Bridges NaN gaps as in-range.

    To keep the recovery-time population consistent with `episode_durations`
    (which drops <15-min crossings), a dip must stay below `low_thresh` for at
    least `min_minutes` before it is treated as an episode worth timing. An
    episode that never re-crosses `normal_thresh` before the record ends is
    excluded rather than recorded as an (uncapped) end-of-record 'recovery',
    which would conflate 'never recovered' with 'slow recovery'."""
    bg = bg.copy()
    bg[np.isnan(bg)] = 120.0
    out = []
    n = len(bg)
    min_samples = max(1, int(round(min_minutes / step_min)))
    i = 0
    while i < n:
        if bg[i] < low_thresh:
            # measure how long BG stays below low_thresh from here
            k = i
            while k < n and bg[k] < low_thresh:
                k += 1
            below_run = k - i
            j = i
            while j < n and bg[j] < normal_thresh:
                j += 1
            if below_run >= min_samples and j < n:
                out.append((j - i) * step_min)
            i = j if j > i else i + 1
        else:
            i += 1
    return out


def jensen_shannon(p, q):
    """JS divergence between two histograms (added to small ε to avoid log(0))."""
    p = np.asarray(p, dtype=float) + 1e-12
    q = np.asarray(q, dtype=float) + 1e-12
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m))))


# ============================================================================
# Dataset assembly
# ============================================================================
def assemble_cohort(name, items, regularize_fn, step_min):
    """Per-record dict + pooled arrays."""
    per = []
    pooled = []
    series = []  # per-record regularised bg (with NaN gaps) for extended stats
    diurnals_mean = []
    diurnals_std = []
    diurnals_median = []
    diurnals_p25 = []
    diurnals_p75 = []
    weekday_hour = defaultdict(list)
    hypo_durs = []
    hyper_durs = []
    severe_hypo_durs = []
    sev_hyper_durs = []
    recov_times = []
    lags_min = [step_min, 15, 30, 60, 120, 240, 480, 720, 1440]
    pooled_acf = {L: [] for L in lags_min}
    delta_pooled = []
    for rid, rows in items:
        t, bg = regularize_fn(rows)
        if len(bg) < 200:
            continue
        bg_clean = bg[~np.isnan(bg)]
        rec = {"rid": str(rid)}
        rec.update(central_moments(bg))
        rec.update(percentile_row(bg))
        rec.update(time_in_ranges(bg))
        lbgi, hbgi = kovatchev_risk(bg)
        rec["LBGI"] = lbgi
        rec["HBGI"] = hbgi
        rec["GMI"] = float(3.31 + 0.02392 * np.mean(bg_clean)) if len(bg_clean) else float("nan")
        rec["j_index"] = j_index(bg)
        rec["m_value"] = m_value(bg)
        rec["mage"] = mage(bg, step_min=step_min)
        rec["conga_1h"] = conga(bg, 1, step_min)
        rec["conga_4h"] = conga(bg, 4, step_min)
        rec["modd"] = modd(t, bg, step_min)
        # Observed (non-NaN) coverage, NOT calendar span: NaN-bridged gap cells
        # carry no episodes (the numerator only accrues in real time), so
        # counting them as exposure deflates every per-day rate for the gappy
        # real cohorts (~8% Ohio, ~1.5% AZT1D) while the gapless sim stays exact.
        days = (int(np.sum(~np.isnan(bg))) * step_min) / (60 * 24)
        h = episode_durations(bg, 70, True, step_min)
        H = episode_durations(bg, 180, False, step_min)
        sh = episode_durations(bg, 54, True, step_min)
        SH = episode_durations(bg, 250, False, step_min)
        rec["days"] = float(days)
        rec["hypo_count_per_day"] = len(h) / days if days else 0
        rec["hyper_count_per_day"] = len(H) / days if days else 0
        rec["severe_hypo_count_per_day"] = len(sh) / days if days else 0
        rec["severe_hyper_count_per_day"] = len(SH) / days if days else 0
        rec["hypo_median_min"] = float(np.median(h)) if h else 0.0
        rec["hypo_p90_min"] = float(np.percentile(h, 90)) if h else 0.0
        rec["hyper_median_min"] = float(np.median(H)) if H else 0.0
        rec["hyper_p90_min"] = float(np.percentile(H, 90)) if H else 0.0
        rec.update({"lag_acf": autocorr_lags(bg, step_min, lags_min)})
        rec["sample_entropy"] = sample_entropy(bg, step_min=step_min)
        diff_rec = np.diff(bg)
        diff_rec = diff_rec[~np.isnan(diff_rec)]
        rec["delta_std"] = float(np.std(diff_rec)) if len(diff_rec) else float("nan")
        per.append(rec)
        pooled.append(bg_clean)
        series.append(bg)
        for L in lags_min:
            v = rec["lag_acf"][L]
            if not np.isnan(v):
                pooled_acf[L].append(v)
        diff = np.diff(bg)
        diff = diff[~np.isnan(diff)]
        delta_pooled.append(diff)
        # diurnal mean + std
        by_hour = defaultdict(list)
        for ts, v in zip(t, bg):
            if not np.isnan(v):
                by_hour[ts.hour].append(v)
                weekday_hour[(ts.weekday(), ts.hour)].append(v)
        m24 = np.array([np.mean(by_hour[h]) if by_hour[h] else np.nan for h in range(24)])
        s24 = np.array([np.std(by_hour[h]) if by_hour[h] else np.nan for h in range(24)])
        med24 = np.array([np.median(by_hour[h]) if by_hour[h] else np.nan for h in range(24)])
        p25_24 = np.array([np.percentile(by_hour[h], 25) if by_hour[h] else np.nan for h in range(24)])
        p75_24 = np.array([np.percentile(by_hour[h], 75) if by_hour[h] else np.nan for h in range(24)])
        diurnals_mean.append(m24)
        diurnals_std.append(s24)
        diurnals_median.append(med24)
        diurnals_p25.append(p25_24)
        diurnals_p75.append(p75_24)
        hypo_durs.extend(h)
        hyper_durs.extend(H)
        severe_hypo_durs.extend(sh)
        sev_hyper_durs.extend(SH)
        recov_times.extend(episode_recovery_time(bg, step_min=step_min))
    # Weekday × hour 7×24 matrices (mean and median across pooled samples per cell)
    wd_grid = np.full((7, 24), np.nan)
    wd_grid_median = np.full((7, 24), np.nan)
    for (wd, h), arr in weekday_hour.items():
        wd_grid[wd, h] = float(np.mean(arr))
        wd_grid_median[wd, h] = float(np.median(arr))
    cohort = {
        "name": name,
        "step_min": step_min,
        "per": per,
        "pooled_bg": np.concatenate(pooled) if pooled else np.array([]),
        "pooled_delta": np.concatenate(delta_pooled) if delta_pooled else np.array([]),
        "diurnal_mean": np.nanmean(np.stack(diurnals_mean), axis=0) if diurnals_mean else np.full(24, np.nan),
        "diurnal_std": np.nanmean(np.stack(diurnals_std), axis=0) if diurnals_std else np.full(24, np.nan),
        "diurnal_median": np.nanmedian(np.stack(diurnals_median), axis=0) if diurnals_median else np.full(24, np.nan),
        "diurnal_p25": np.nanmedian(np.stack(diurnals_p25), axis=0) if diurnals_p25 else np.full(24, np.nan),
        "diurnal_p75": np.nanmedian(np.stack(diurnals_p75), axis=0) if diurnals_p75 else np.full(24, np.nan),
        "wd_grid": wd_grid,
        "wd_grid_median": wd_grid_median,
        "pooled_acf": {L: (float(np.mean(v)) if v else float("nan")) for L, v in pooled_acf.items()},
        "lags_min": lags_min,
        "hypo_durs": hypo_durs,
        "hyper_durs": hyper_durs,
        "severe_hypo_durs": severe_hypo_durs,
        "severe_hyper_durs": sev_hyper_durs,
        "recov_times": recov_times,
        "series": series,
        "series_step": step_min,
    }
    return cohort


def assemble_sim(n_seeds=30, days=70, warmup_h=24):
    """Run the simulator once per seed and capture every output channel.

    Returns a tuple `(items, raw_runs)` where:
      - `items` is the list of (seed_id, [(ts, bg_observed), ...]) tuples used
        by `assemble_cohort` to produce all CGM-only statistics.
      - `raw_runs` is the per-seed dict of `generate_hours()` output arrays
        (`bg_observed`, `total_carb`, `basal_insulin`, `bolus_insulin`, ...),
        used by `sim_event_summary` so the simulator is only exercised once.
    """
    print(f"Running T1DMSIM: {n_seeds} seeds × {days}d ({warmup_h}h warmup)…")
    items = []
    raw_runs = []
    for seed in range(n_seeds):
        s = T1DMSimulator(seed=seed, initial_bg=120.0)
        s.generate_hours(warmup_h)
        d = s.generate_hours(days * 24)
        bg = np.asarray(d["bg_observed"], dtype=float)
        # Advance the calendar origin by the warm-up so the harness weekday
        # matches the simulator's internal day-of-week. The sim starts on a
        # Monday (SIMULATION_START_DAY_OF_WEEK=0) and 2024-01-01 is a Monday;
        # the discarded warm-up advances the internal clock, so without this
        # shift the whole weekday×hour grid is rotated one day relative to the
        # real (true-calendar) cohorts.
        t0 = datetime(2024, 1, 1) + timedelta(hours=warmup_h)
        rows = [(t0 + timedelta(minutes=5 * i), float(bg[i])) for i in range(len(bg))]
        items.append((str(seed), rows))
        raw_runs.append((str(seed), d))
    return items, raw_runs


def trivial_regularize_5min(rows):
    """Sim rows are already on a 5-min grid; pass through."""
    times = np.array([r[0] for r in rows])
    vals = np.array([r[1] for r in rows], dtype=float)
    return times, vals


# ============================================================================
# AZT1D-only event panel (basal rates, bolus type breakdown, carbs)
# ============================================================================
def azt1d_event_summary(events_by_subject):
    """Aggregate per-subject insulin / carb / device-mode statistics.

    Returns a dict suitable for both the markdown table and the bar chart:
        {
            "per_subject": [...],
            "pooled": {basal stats, bolus split, carb stats, device-mode %},
        }
    """
    import numpy as _np
    per = []
    all_basal = []
    bolus_type_counts = defaultdict(int)
    correction_units = 0.0
    food_units = 0.0
    total_bolus_units = 0.0
    n_bolus_events = 0
    n_correction_events = 0  # standalone "correction-only" boluses
    n_meal_events = 0
    all_carbs = []
    device_mode_minutes = defaultdict(int)
    total_minutes = 0
    total_days = 0.0
    # Several AZT1D subjects have a small fraction of clearly bogus basal-rate
    # entries (e.g. 1000+ U/hr — likely an OCR or unit-encoding artefact in the
    # original PDF-to-CSV extraction described in the manuscript). Cap at a
    # physiologically plausible upper bound before any aggregation; the
    # discarded fraction is recorded below.
    BASAL_CAP_U_PER_HR = 10.0
    discarded_basal = 0
    total_basal = 0
    for sub, df in events_by_subject.items():
        if df.empty:
            continue
        n_rows = len(df)
        total_minutes += n_rows * 5
        days = n_rows * 5 / (60 * 24)
        total_days += days
        # Basal rate: rows where 'Basal' is set carry the active pump rate in U/hr.
        basal = df["Basal"].dropna().to_numpy()
        total_basal += len(basal)
        discarded_basal += int((basal > BASAL_CAP_U_PER_HR).sum())
        basal = basal[(basal >= 0) & (basal <= BASAL_CAP_U_PER_HR)]
        all_basal.extend(basal.tolist())
        # Bolus rows: any non-null BolusType. The pump exposes several
        # categories — split user-driven boluses (Standard, Standard/Correction
        # and the BLE variants) from AID-driven `Automatic Bolus/Correction`,
        # because the simulator models MDI long-acting basal + per-meal user
        # boluses with no AID counterpart, so mixing both would skew the
        # bolus-events-per-day comparison by an order of magnitude.
        bdf = df.dropna(subset=["BolusType"])
        bt_str = bdf["BolusType"].astype(str)
        is_automatic = bt_str.str.contains("Automatic", case=False, regex=False)
        # Drop pump-internal "0" placeholder rows and explicit AID auto-boluses
        # from the user-bolus count, but still log their existence.
        is_zero_row = bt_str.str.strip().isin(["0", "0.0"])
        user_mask = ~(is_automatic | is_zero_row)
        # Skip the literal "0" placeholder (Subject 8 reports ~7800 rows with
        # BolusType == "0", a known extraction artefact in the published
        # CSV — meaningless category, not a real bolus).
        for bt, cnt in bt_str.value_counts().items():
            label = str(bt).strip()
            if label in ("0", "0.0", "nan", ""):
                continue
            bolus_type_counts[label] += int(cnt)
        # Restrict the per-day / per-meal / unit-share computations to
        # user-initiated boluses for the simulator-comparable view.
        ubdf = bdf[user_mask]
        n_bolus_events += len(ubdf)
        tot = ubdf["TotalBolusInsulinDelivered"].fillna(0.0).to_numpy()
        corr = ubdf["CorrectionDelivered"].fillna(0.0).to_numpy()
        food = ubdf["FoodDelivered"].fillna(0.0).to_numpy()
        correction_units += float(corr.sum())
        food_units += float(food.sum())
        total_bolus_units += float(tot.sum())
        is_meal = food > 1e-6
        is_corr_only = (~is_meal) & (corr > 1e-6)
        n_meal_events += int(is_meal.sum())
        n_correction_events += int(is_corr_only.sum())
        carbs = ubdf.loc[is_meal, "CarbSize"].dropna().to_numpy()
        all_carbs.extend(carbs.tolist())
        # Device mode minutes (each row covers 5 minutes). Several subjects'
        # CSVs contain typos ("sleepsleep") or numeric-coerced placeholders
        # ("0"); fold those into the canonical sleep/exercise/regular buckets.
        dm = df["DeviceMode"].fillna("regular").astype(str).str.strip().str.lower()
        dm = dm.replace({"": "regular", "nan": "regular", "0": "regular",
                         "sleepsleep": "sleep", "exerciseexercise": "exercise"})
        for mode, cnt in dm.value_counts().items():
            key = str(mode)
            if "sleep" in key:
                key = "sleep"
            elif "exercise" in key:
                key = "exercise"
            elif key not in ("regular",):
                key = "regular"
            device_mode_minutes[key] += int(cnt) * 5
        # Per-subject summary — basal arr was already capped above; bolus
        # counts here are the *user-initiated* subset (AID auto-boluses
        # excluded so the per-day counts are comparable to the simulator).
        per.append({
            "sub": str(sub),
            "days": float(days),
            "basal_mean_U_per_hr": float(_np.mean(basal)) if len(basal) else float("nan"),
            "basal_median_U_per_hr": float(_np.median(basal)) if len(basal) else float("nan"),
            "basal_cv_pct": float(100 * _np.std(basal) / _np.mean(basal))
                            if len(basal) and _np.mean(basal) > 0 else float("nan"),
            "boluses_per_day": float(len(ubdf) / days) if days else 0.0,
            "meal_boluses_per_day": float(int(is_meal.sum()) / days) if days else 0.0,
            "correction_boluses_per_day": float(int(is_corr_only.sum()) / days)
                                          if days else 0.0,
            "carbs_per_day_g": float(carbs.sum() / days) if days else 0.0,
            "mean_carb_per_meal_g": float(_np.mean(carbs)) if len(carbs) else float("nan"),
            "correction_unit_share_pct": float(100 * corr.sum() / tot.sum())
                                          if tot.sum() > 0 else float("nan"),
            # Total daily insulin: basal U/hr × 24h + total bolus units. We
            # estimate basal contribution from the *mean* sampled rate × 24h
            # (basal samples are uniform 5-min snapshots, so mean × 24 is a
            # well-defined daily-equivalent).
            "total_insulin_U_per_day": float(
                (_np.mean(basal) * 24 if len(basal) else 0.0) + tot.sum() / days)
                if days else 0.0,
        })
    pooled = {
        "n_subjects": len(per),
        "total_days": float(total_days),
        "basal_pooled_mean": float(_np.mean(all_basal)) if all_basal else float("nan"),
        "basal_pooled_median": float(_np.median(all_basal)) if all_basal else float("nan"),
        "basal_pooled_p10": float(_np.percentile(all_basal, 10)) if all_basal else float("nan"),
        "basal_pooled_p90": float(_np.percentile(all_basal, 90)) if all_basal else float("nan"),
        "boluses_per_day_mean": float(_np.mean([r["boluses_per_day"] for r in per])) if per else 0.0,
        "meal_boluses_per_day_mean": float(_np.mean([r["meal_boluses_per_day"] for r in per])) if per else 0.0,
        "correction_boluses_per_day_mean": float(_np.mean([r["correction_boluses_per_day"] for r in per])) if per else 0.0,
        "carbs_per_day_mean": float(_np.mean([r["carbs_per_day_g"] for r in per])) if per else 0.0,
        "mean_carb_per_meal_mean": float(_np.mean([r["mean_carb_per_meal_g"]
                                                    for r in per
                                                    if not _np.isnan(r["mean_carb_per_meal_g"])])) if per else 0.0,
        "correction_unit_share_pct": float(100 * correction_units / total_bolus_units)
                                      if total_bolus_units > 0 else float("nan"),
        "food_unit_share_pct": float(100 * food_units / total_bolus_units)
                                if total_bolus_units > 0 else float("nan"),
        "total_insulin_per_day_mean": float(_np.mean([r["total_insulin_U_per_day"] for r in per])) if per else 0.0,
        "bolus_type_counts": dict(bolus_type_counts),
        "device_mode_minutes": dict(device_mode_minutes),
        "device_mode_pct": {k: 100.0 * v / total_minutes
                            for k, v in device_mode_minutes.items()}
                            if total_minutes else {},
        "basal_capped_pct": (100.0 * discarded_basal / total_basal
                              if total_basal else 0.0),
        "basal_cap_threshold": BASAL_CAP_U_PER_HR,
    }
    return {"per_subject": per, "pooled": pooled,
            "basal_values": all_basal, "carb_values": all_carbs}


def sim_event_summary(raw_runs):
    """Per-seed insulin / carb stats from already-generated sim runs.

    Consumes the `raw_runs` second tuple element returned by `assemble_sim`
    so the simulator is only invoked once per seed.

    We deliberately report only **integrated daily totals** (carbs/day,
    insulin/day) and **basal-rate distributions** — quantities that are
    directly comparable to AZT1D. Per-meal carb size and per-bolus event
    counts are NOT computed for the simulator because its
    `bolus_insulin` / `total_carb` channels expose active-PK / active-
    absorption levels rather than discrete injection events, and any
    threshold-based clustering merges events whose curves overlap. The
    AZT1D-side per-meal / per-event numbers stay AZT1D-only in the report.
    """
    import numpy as _np
    per = []
    all_basal = []
    for seed, d in raw_runs:
        carbs_arr = _np.asarray(d.get("total_carb", []), dtype=float)
        basal_arr = _np.asarray(d.get("basal_insulin", []), dtype=float)
        bolus_arr = _np.asarray(d.get("bolus_insulin", []), dtype=float)
        # basal_insulin / bolus_insulin are per-step (5 min) PK-curve samples,
        # i.e. units of insulin reaching circulation in this step. Their sum
        # over a full DIA equals the original injection size, so integrating
        # over the whole run gives total delivered units. To project the
        # per-step amount into a U/hr "rate" comparable to AZT1D's hourly
        # basal column, multiply by 12 (12 × 5min = 60min).
        basal_rate = basal_arr * 12.0
        n_steps = len(carbs_arr)
        d_days = n_steps * 5 / (60 * 24)
        per.append({
            "sub": str(seed),
            "days": float(d_days),
            "basal_mean_U_per_hr": float(_np.mean(basal_rate)),
            "basal_median_U_per_hr": float(_np.median(basal_rate)),
            "basal_cv_pct": float(100 * _np.std(basal_rate) / _np.mean(basal_rate))
                            if _np.mean(basal_rate) > 0 else float("nan"),
            "carbs_per_day_g": float(carbs_arr.sum() / d_days) if d_days else 0.0,
            "total_insulin_U_per_day": float((basal_arr.sum() + bolus_arr.sum()) / d_days)
                                       if d_days else 0.0,
        })
        all_basal.extend(basal_rate.tolist())
    pooled = {
        "n_subjects": len(per),
        "total_days": float(sum(r["days"] for r in per)),
        "basal_pooled_mean": float(_np.mean(all_basal)) if all_basal else float("nan"),
        "basal_pooled_median": float(_np.median(all_basal)) if all_basal else float("nan"),
        "basal_pooled_p10": float(_np.percentile(all_basal, 10)) if all_basal else float("nan"),
        "basal_pooled_p90": float(_np.percentile(all_basal, 90)) if all_basal else float("nan"),
        "carbs_per_day_mean": float(_np.mean([r["carbs_per_day_g"] for r in per])) if per else 0.0,
        "total_insulin_per_day_mean": float(_np.mean([r["total_insulin_U_per_day"] for r in per])) if per else 0.0,
    }
    return {"per_subject": per, "pooled": pooled,
            "basal_values": all_basal}


def fig_azt1d_events(az_stats, sim_stats, path):
    """4-panel panel comparing AZT1D and Sim insulin/carb behaviour."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.2))

    # (1) Basal-rate distribution (U/hr)
    ax = axes[0, 0]
    bins = np.linspace(0.0, 4.0, 41)
    az_basal = np.asarray(az_stats["basal_values"])
    sim_basal = np.asarray(sim_stats["basal_values"])
    if len(az_basal):
        ax.hist(az_basal, bins=bins, density=True, histtype="step",
                lw=2.2, color=COL["AZT1D"], label="AZT1D AID basal rate")
    if len(sim_basal):
        ax.hist(sim_basal, bins=bins, density=True, histtype="step",
                lw=2.2, color=COL["Sim"], label="T1DMSIM basal-channel rate")
    ax.set_xlabel("basal insulin (U/hr)")
    ax.set_ylabel("density")
    ax.set_title("Basal-rate distribution", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    # (2) Per-subject daily insulin / carb totals — integrated quantities,
    # comparable across data sources.
    ax = axes[0, 1]
    az_pool = az_stats["pooled"]
    sim_pool = sim_stats["pooled"]
    cats = ["carbs / day (g)", "total insulin / day (U)"]
    az_vals = [az_pool["carbs_per_day_mean"],
               az_pool["total_insulin_per_day_mean"]]
    sim_vals = [sim_pool["carbs_per_day_mean"],
                sim_pool["total_insulin_per_day_mean"]]
    x = np.arange(len(cats))
    w = 0.38
    ax.bar(x - w / 2, az_vals, w, color=COL["AZT1D"], label="AZT1D")
    ax.bar(x + w / 2, sim_vals, w, color=COL["Sim"], label="T1DMSIM")
    ymax = max(max(az_vals), max(sim_vals)) * 1.18
    for i, (a, b) in enumerate(zip(az_vals, sim_vals)):
        ax.text(i - w / 2, a + ymax * 0.015, f"{a:.0f}", ha="center", fontsize=9)
        ax.text(i + w / 2, b + ymax * 0.015, f"{b:.0f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=9)
    ax.set_ylim(0, ymax)
    ax.set_ylabel("per-subject mean")
    ax.set_title("Daily intake totals (integrated)", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    # (3) AZT1D-only — per-meal carb-size distribution. The simulator's
    # `total_carb` channel cannot be cleanly split into discrete meal events
    # (overlapping gamma absorption tails), so this is reported as an AZT1D
    # reference distribution rather than a head-to-head comparison.
    ax = axes[1, 0]
    bins = np.linspace(0, 120, 31)
    az_carbs = np.asarray(az_stats.get("carb_values", []))
    if len(az_carbs):
        ax.hist(az_carbs, bins=bins, density=True, histtype="step",
                lw=2.2, color=COL["AZT1D"],
                label=f"AZT1D meals (n={len(az_carbs):,})")
    ax.set_xlabel("carbohydrates per meal (g)")
    ax.set_ylabel("density")
    ax.set_title("Per-meal carb distribution (AZT1D only)",
                 fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    # (4) AZT1D device-mode time share
    ax = axes[1, 1]
    dm_pct = az_pool.get("device_mode_pct", {})
    # Collapse anything that isn't sleep/exercise into "regular".
    dm_norm = {"regular": 0.0, "sleep": 0.0, "exercise": 0.0}
    for k, v in dm_pct.items():
        if k == "sleep":
            dm_norm["sleep"] += v
        elif k == "exercise":
            dm_norm["exercise"] += v
        else:
            dm_norm["regular"] += v
    labels = list(dm_norm.keys())
    vals = list(dm_norm.values())
    cs = ["#86C893", "#5B8DB8", "#E08550"]
    x = np.arange(len(labels))
    bars = ax.bar(x, vals, color=cs, edgecolor="black", linewidth=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5,
                f"{v:.1f}%", ha="center", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("% of total recorded minutes")
    ax.set_title("AZT1D device-mode time share", fontweight="bold")
    ax.set_ylim(0, max(vals) * 1.25 if vals else 1)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("AZT1D vs T1DMSIM — insulin / carb / device-mode behaviour",
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ============================================================================
# Unexplained-excursion analysis (§6.3)
# ============================================================================
# An excursion is a >= EXC_AMP mg/dL monotone swing (MAGE-style turning-point
# detection on lightly smoothed CGM). It is "explained" if a logged meal
# precedes a rise / a logged bolus or exercise precedes a fall; otherwise
# "unexplained" — an unlogged event or a genuinely endogenous movement (dawn
# phenomenon, post-hypo rebound, stress, illness, sensor artefact). The
# simulator is held to the identical test, its meal / bolus / exercise events
# taken as the rising edges of the corresponding factor channels.
EXC_AMP = 40.0          # mg/dL minimum excursion amplitude
EXC_REVERSAL = 15.0     # mg/dL turning-point confirmation threshold
EXC_CARB_MIN = 10.0     # g minimum logged carbs to count as a meal cause
EXC_INS_MIN = 0.5       # U minimum logged bolus to count as an insulin cause
EXC_RISE_PRE, EXC_RISE_POST = 60.0, 15.0   # min: meal window around a rise
EXC_FALL_PRE, EXC_FALL_POST = 90.0, 15.0   # min: bolus window around a fall
EXC_EX_PRE, EXC_EX_POST = 60.0, 15.0       # min: exercise window around a fall
EXC_GREEN = "#1B5E20"
EXC_CRIMSON = "#B0143C"
EXC_GREY = "#8a8a8a"


def _exc_smooth(x, w=3):
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="same")


def _exc_zigzag(seg, reversal):
    """Alternating extrema [(idx, val, kind)] within a NaN-free segment."""
    m = len(seg)
    if m < 3:
        return []
    ext = []
    hi_i = lo_i = 0
    hi_v = lo_v = seg[0]
    trend = 0
    for k in range(1, m):
        v = seg[k]
        if v > hi_v:
            hi_v, hi_i = v, k
        if v < lo_v:
            lo_v, lo_i = v, k
        if trend >= 0 and hi_v - v >= reversal:
            ext.append((hi_i, hi_v, "peak"))
            trend = -1
            lo_v, lo_i = v, k
            hi_v, hi_i = v, k
        elif trend <= 0 and v - lo_v >= reversal:
            ext.append((lo_i, lo_v, "trough"))
            trend = 1
            hi_v, hi_i = v, k
            lo_v, lo_i = v, k
    return ext


def detect_excursions(bg, amp=EXC_AMP, reversal=EXC_REVERSAL):
    """[(onset_idx, peak_idx, kind, amplitude)] over NaN-bridged segments."""
    n = len(bg)
    out = []
    i = 0
    while i < n:
        if np.isnan(bg[i]):
            i += 1
            continue
        j = i
        while j < n and not np.isnan(bg[j]):
            j += 1
        ext = _exc_zigzag(_exc_smooth(bg[i:j].astype(float)), reversal)
        for a, b in zip(ext, ext[1:]):
            amplitude = abs(b[1] - a[1])
            if amplitude >= amp:
                kind = "rise" if b[1] > a[1] else "fall"
                out.append((i + a[0], i + b[0], kind, amplitude))
        i = j
    return out


def _exc_any_in(sorted_min, lo, hi):
    if len(sorted_min) == 0:
        return False
    return np.searchsorted(sorted_min, hi, "right") > \
        np.searchsorted(sorted_min, lo, "left")


def _exc_attribute(onset_min, kind, meals, boluses, exers):
    if kind == "rise":
        return _exc_any_in(meals, onset_min - EXC_RISE_PRE,
                           onset_min + EXC_RISE_POST)
    ok = _exc_any_in(boluses, onset_min - EXC_FALL_PRE,
                     onset_min + EXC_FALL_POST)
    if not ok and len(exers):
        ok = _exc_any_in(exers, onset_min - EXC_EX_PRE,
                         onset_min + EXC_EX_POST)
    return ok


def load_ohio_events():
    """{pid: (meals[(dt,carbs)], boluses[(dt,units)], exercise[dt])}."""
    out = defaultdict(lambda: ([], [], []))
    if not os.path.isdir(OHIO_DIR):
        return out
    for fn in os.listdir(OHIO_DIR):
        if not fn.endswith(".xml"):
            continue
        pid = fn.split("-")[0]
        try:
            root = ET.parse(os.path.join(OHIO_DIR, fn)).getroot()
        except ET.ParseError:
            continue
        m, b, e = out[pid]
        for tag, lst in (("meal", m), ("bolus", b), ("exercise", e)):
            for node in root.iter(tag):
                for ev in node.iter("event"):
                    ts = ev.get("ts") or ev.get("ts_begin")
                    if not ts:
                        continue
                    try:
                        dt = datetime.strptime(ts, "%d-%m-%Y %H:%M:%S")
                    except ValueError:
                        continue
                    if tag == "meal":
                        c = float(ev.get("carbs") or 0)
                        if c >= EXC_CARB_MIN:
                            lst.append((dt, c))
                    elif tag == "bolus":
                        d = float(ev.get("dose") or 0)
                        if d >= EXC_INS_MIN:
                            lst.append((dt, d))
                    else:
                        lst.append(dt)
        out[pid] = (m, b, e)
    return out


def azt1d_event_lists(df):
    """(meals, boluses, exercise) lists from one AZT1D event DataFrame."""
    meals, boluses = [], []
    t = df["EventDateTime"]
    carb = df.get("CarbSize")
    food = df.get("FoodDelivered")
    bol = df.get("TotalBolusInsulinDelivered")
    for i in range(len(df)):
        ts = t.iloc[i]
        if ts is None or (isinstance(ts, float) and np.isnan(ts)):
            continue
        ts = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
        c = 0.0
        if carb is not None and not np.isnan(carb.iloc[i]):
            c = float(carb.iloc[i])
        elif food is not None and not np.isnan(food.iloc[i]):
            c = float(food.iloc[i])
        if c >= EXC_CARB_MIN:
            meals.append((ts, c))
        if bol is not None and not np.isnan(bol.iloc[i]) and bol.iloc[i] >= EXC_INS_MIN:
            boluses.append((ts, float(bol.iloc[i])))
    return meals, boluses, []


def _exc_onsets(channel, eps, refractory=4):
    """Rising-edge onsets of a simulator factor channel."""
    on = []
    active = False
    last = -9999
    for k, v in enumerate(channel):
        if v >= eps:
            if not active and k - last >= refractory:
                on.append(k)
                last = k
            active = True
        else:
            active = False
    return on


def _exc_to_minutes(times):
    base = times[0]
    return np.array([(t - base).total_seconds() / 60.0 for t in times])


def _exc_process_record(times, bg, meals, boluses, exers):
    tmin = _exc_to_minutes(times)
    meal_min = np.array(sorted((m[0] - times[0]).total_seconds() / 60.0
                               for m in meals))
    bol_min = np.array(sorted((b[0] - times[0]).total_seconds() / 60.0
                              for b in boluses))
    ex_min = np.array(sorted((e - times[0]).total_seconds() / 60.0
                             for e in exers))
    recs = []
    for onset_i, peak_i, kind, amp in detect_excursions(bg):
        om = tmin[onset_i]
        recs.append({
            "onset_i": onset_i, "peak_i": peak_i, "kind": kind, "amp": amp,
            "explained": _exc_attribute(om, kind, meal_min, bol_min, ex_min),
            "hour": times[onset_i].hour + times[onset_i].minute / 60.0,
        })
    return recs


def _exc_censored_dbg_sd(bg, excs, step_min):
    mask = np.zeros(len(bg), dtype=bool)
    for e in excs:
        if not e["explained"]:
            mask[e["onset_i"]:e["peak_i"] + 1] = True
    d = np.diff(bg)
    keep = ~np.isnan(d)
    full = float(np.std(d[keep])) if keep.any() else float("nan")
    cok = keep & ~mask[:-1] & ~mask[1:]
    cens = float(np.std(d[cok])) if cok.any() else float("nan")
    return full, cens


def unexplained_cohort_real(name, cgm_items, events_fn, step_min=5):
    excs_all = []
    records = []
    for rid, rows in cgm_items:
        t, bg = regularize_bg(rows)
        if len(bg) < 200:
            continue
        meals, boluses, exers = events_fn(rid)
        excs = _exc_process_record(t, bg, meals, boluses, exers)
        for e in excs:
            e["rid"] = str(rid)
        excs_all.extend(excs)
        records.append((str(rid), t, bg, meals, boluses, excs,
                        len(bg) * step_min / 1440.0))
    return {"name": name, "excs": excs_all, "records": records,
            "step_min": step_min}


def unexplained_cohort_sim(raw_runs):
    """Reuse assemble_sim's raw runs — the simulator is not re-exercised."""
    excs_all = []
    records = []
    base = datetime(2024, 1, 1)
    for seed, d in raw_runs:
        bg = np.asarray(d["bg_observed"], dtype=float)
        carb = np.asarray(d["total_carb"], dtype=float)
        bol = np.asarray(d["bolus_insulin"], dtype=float)
        exe = np.asarray(d.get("total_exercise", []), dtype=float)
        n = len(bg)
        times = [base + timedelta(minutes=5 * i) for i in range(n)]
        meals = [(times[k], 0.0) for k in _exc_onsets(carb, 0.05)]
        boluses = [(times[k], 0.0) for k in _exc_onsets(bol, 0.005)]
        exers = [times[k] for k in _exc_onsets(exe, 0.05)] if len(exe) else []
        excs = _exc_process_record(times, bg, meals, boluses, exers)
        for e in excs:
            e["rid"] = f"seed{seed}"
        excs_all.extend(excs)
        records.append((f"seed{seed}", times, bg, meals, boluses, excs,
                        n * 5 / 1440.0))
    return {"name": "T1DMSIM", "excs": excs_all, "records": records,
            "step_min": 5}


def unexplained_summarize(coh):
    excs = coh["excs"]
    rises = [e for e in excs if e["kind"] == "rise"]
    falls = [e for e in excs if e["kind"] == "fall"]
    total_days = sum(r[6] for r in coh["records"])

    def frac_un(lst):
        return 100.0 * np.mean([not e["explained"] for e in lst]) if lst else 0.0

    def load(expl):
        v = [e["amp"] for e in excs if e["explained"] == expl]
        return sum(v) / total_days if total_days else 0.0

    def rate(expl):
        v = [e for e in excs if e["explained"] == expl]
        return len(v) / total_days if total_days else 0.0

    def amp(expl):
        v = [e["amp"] for e in excs if e["explained"] == expl]
        return float(np.median(v)) if v else float("nan")

    full_sd, cens_sd = [], []
    for (_rid, _t, bg, _m, _b, excs_r, _d) in coh["records"]:
        f, c = _exc_censored_dbg_sd(bg, excs_r, coh["step_min"])
        if not np.isnan(f):
            full_sd.append(f)
        if not np.isnan(c):
            cens_sd.append(c)
    return {
        "name": coh["name"], "n_exc": len(excs), "n_rise": len(rises),
        "n_fall": len(falls), "days": float(total_days),
        "rise_unexpl_pct": frac_un(rises), "fall_unexpl_pct": frac_un(falls),
        "all_unexpl_pct": frac_un(excs),
        "rate_expl": rate(True), "rate_unexpl": rate(False),
        "load_expl": load(True), "load_unexpl": load(False),
        "amp_expl": amp(True), "amp_unexpl": amp(False),
        "dbg_sd_full": float(np.mean(full_sd)) if full_sd else float("nan"),
        "dbg_sd_cens": float(np.mean(cens_sd)) if cens_sd else float("nan"),
    }


def fig_unexplained_summary(summ, real_pool, path):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    names = [s["name"] for s in summ]
    x = np.arange(len(names))

    w = 0.38
    ax[0].bar(x - w / 2, [s["rise_unexpl_pct"] for s in summ], w,
              label="rises (no logged meal)", color="#c44e52")
    ax[0].bar(x + w / 2, [s["fall_unexpl_pct"] for s in summ], w,
              label="falls (no logged insulin/exercise)", color="#4c72b0")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(names)
    ax[0].set_ylabel("% of excursions unexplained")
    ax[0].set_title("(a) Unexplained fraction by direction")
    ax[0].legend(fontsize=8, frameon=False)
    ax[0].grid(axis="y", alpha=0.3)

    for label, excs, days, col in real_pool:
        un = [e["hour"] for e in excs if not e["explained"]]
        h, edges = np.histogram(un, bins=np.arange(0, 25, 2))
        centers = (edges[:-1] + edges[1:]) / 2
        ax[1].plot(centers, h / days, marker="o", ms=3, label=label, color=col)
    ax[1].set_xlabel("hour of day")
    ax[1].set_ylabel("unexplained excursions / day")
    ax[1].set_title("(b) When unexplained excursions occur")
    ax[1].set_xticks(range(0, 25, 4))
    ax[1].legend(fontsize=8, frameon=False)
    ax[1].grid(alpha=0.3)

    le = [s["load_expl"] for s in summ]
    lu = [s["load_unexpl"] for s in summ]
    ax[2].bar(x, le, 0.55, label="explained", color="#5a5a5a")
    ax[2].bar(x, lu, 0.55, bottom=le, label="unexplained",
              color=EXC_CRIMSON, hatch="//", edgecolor="white")
    ax[2].set_xticks(x)
    ax[2].set_xticklabels(names)
    ax[2].set_ylabel("excursion load (Σ amplitude, mg/dL/day)")
    ax[2].set_title("(c) Variability load: explained vs unexplained")
    ax[2].legend(fontsize=8, frameon=False)
    ax[2].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_unexplained_gallery(real_cohorts, path):
    from matplotlib.patches import Patch
    import matplotlib.dates as mdates
    picks = []
    for coh in real_cohorts:
        cands = [(rec, e) for rec in coh["records"] for e in rec[5]
                 if not e["explained"] and e["amp"] >= 55]
        cands.sort(key=lambda ce: -ce[1]["amp"])
        seen = set()
        for rec, e in cands:
            if rec[0] in seen:
                continue
            seen.add(rec[0])
            picks.append((coh["name"], rec, e))
            if len(seen) >= 3:
                break
    picks = picks[:6]

    fig, axes = plt.subplots(3, 2, figsize=(13, 10))
    for ax, (cname, rec, e) in zip(axes.ravel(), picks):
        rid, t, bg, meals, boluses, excs, _days = rec
        c = (e["onset_i"] + e["peak_i"]) // 2
        half = int(8 * 60 / 5)
        lo, hi = max(0, c - half), min(len(bg), c + half)
        tt = t[lo:hi]
        ax.plot(tt, bg[lo:hi], color="#222", lw=1.1, zorder=3)
        ax.axhspan(70, 180, color="#2ca02c", alpha=0.07, zorder=0)
        for ee in excs:
            if ee["peak_i"] < lo or ee["onset_i"] > hi:
                continue
            a, b = max(lo, ee["onset_i"]), min(hi - 1, ee["peak_i"])
            col = EXC_CRIMSON if not ee["explained"] else EXC_GREY
            ax.axvspan(t[a], t[b], color=col,
                       alpha=0.22 if not ee["explained"] else 0.12, zorder=1)
        t0, t1 = tt[0], tt[-1]
        for (mt, _c) in meals:
            if t0 <= mt <= t1:
                ax.plot(mt, 30, marker="^", color=EXC_GREEN, ms=7, zorder=4)
        for (bt, _u) in boluses:
            if t0 <= bt <= t1:
                ax.plot(bt, 30, marker="v", color="#1f5fb0", ms=7, zorder=4)
        ax.set_title(f"{cname} · {rid} · unexplained "
                     f"{e['kind']} {e['amp']:.0f} mg/dL", fontsize=9)
        ax.set_ylim(20, 400)
        ax.tick_params(labelsize=7)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    legend = [
        Patch(facecolor=EXC_CRIMSON, alpha=0.3, label="unexplained excursion"),
        Patch(facecolor=EXC_GREY, alpha=0.2, label="explained excursion"),
        plt.Line2D([], [], marker="^", color=EXC_GREEN, ls="",
                   label="logged meal"),
        plt.Line2D([], [], marker="v", color="#1f5fb0", ls="",
                   label="logged bolus"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=4, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Unexplained excursions in real CGM (no logged meal before a "
                 "rise / no logged insulin before a fall)", fontsize=11)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Distribution distance metrics
# ============================================================================
def distribution_distances(a, b, bins=None):
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return {"ks_stat": float("nan"), "ks_p": float("nan"),
                "wasserstein": float("nan"), "js_div": float("nan")}
    ks_stat, ks_p = sps.ks_2samp(a, b)
    w = float(sps.wasserstein_distance(a, b))
    if bins is None:
        # Cover the full plausible support, not just [40,400]. Ohio/AZT1D and
        # the simulator are hard-clamped to [40,400], but Shanghai is
        # uncensored (39.6-475.2); a [40,400] window silently drops ~1.3% of
        # Shanghai's mass — precisely the extreme tail the clamped sim cannot
        # reproduce — flattering every Shanghai JS pair. Wide bins count it.
        bins = np.arange(0, 601, 5)
    ha, _ = np.histogram(a, bins=bins, density=True)
    hb, _ = np.histogram(b, bins=bins, density=True)
    js = jensen_shannon(ha, hb)
    return {"ks_stat": float(ks_stat), "ks_p": float(ks_p),
            "wasserstein": w, "js_div": js}


# ============================================================================
# Aux summaries (per-record deltas, recovery times)
# ============================================================================
def recovery_summary(times):
    """Median / p75 / p90 / p99 / max / n for a recovery-time array (minutes)."""
    if not times:
        return {"n": 0, "median": float("nan"), "p75": float("nan"),
                "p90": float("nan"), "p99": float("nan"), "max": float("nan")}
    arr = np.asarray(times, dtype=float)
    return {
        "n": int(len(arr)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }




# ============================================================================
# Cohort summary (pop mean/std + IQR across per-record stats)
# ============================================================================
def cohort_summary(per):
    if not per:
        return {}
    keys = set()
    for p in per:
        for k, v in p.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                keys.add(k)
    out = {}
    for k in sorted(keys):  # set order is hash-seed dependent; stats.json must be stable
        vals = [p[k] for p in per if k in p and isinstance(p[k], (int, float))
                and not isinstance(p[k], bool) and not np.isnan(p[k])]
        if not vals:
            continue
        v = np.asarray(vals)
        out[k] = {
            "mean": float(np.mean(v)),
            "std": float(np.std(v)),
            "min": float(np.min(v)),
            "p25": float(np.percentile(v, 25)),
            "median": float(np.median(v)),
            "p75": float(np.percentile(v, 75)),
            "max": float(np.max(v)),
            "n": int(len(v)),
        }
    return out


# ============================================================================
# Figures
# ============================================================================
def fig_pdf_pooled(cohorts, path):
    bins = np.arange(30, 401, 5)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.axvspan(70, 180, color="lightgreen", alpha=0.2, label="TIR")
    for n in ORDER:
        d = cohorts[n]["pooled_bg"]
        ax.hist(d, bins=bins, density=True, histtype="step", lw=2.2,
                color=COL[n], label=f"{n} (n={len(d):,})")
    for x in (54, 70, 180, 250):
        ax.axvline(x, color="grey", lw=0.7, ls=":")
    ax.set_xlabel("BG (mg/dL)")
    ax.set_ylabel("density")
    ax.set_title("Pooled CGM-value density")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_cdf_pooled(cohorts, path):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for n in ORDER:
        d = np.sort(cohorts[n]["pooled_bg"])
        if len(d) == 0:
            continue
        cdf = np.arange(1, len(d) + 1) / len(d)
        ax.plot(d, cdf, color=COL[n], lw=2.0, label=n)
    for x in (54, 70, 180, 250):
        ax.axvline(x, color="grey", lw=0.7, ls=":")
    ax.set_xlabel("BG (mg/dL)")
    ax.set_ylabel("empirical CDF")
    ax.set_title("Empirical CDF of CGM values — KS statistic = max vertical gap")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_diurnal_envelope(cohorts, path):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.axhspan(70, 180, color="lightgreen", alpha=0.18)
    h = np.arange(24)
    for n in ORDER:
        m = cohorts[n]["diurnal_mean"]
        s = cohorts[n]["diurnal_std"]
        ax.plot(h, m, color=COL[n], lw=2.2, marker="o", ms=4, label=f"{n} mean")
        ax.fill_between(h, m - s, m + s, color=COL[n], alpha=0.15)
    ax.set_xticks(np.arange(0, 25, 3))
    ax.set_xlabel("hour of day")
    ax.set_ylabel("BG (mg/dL)")
    ax.set_title("Diurnal BG: per-record mean and ±1σ envelope")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_diurnal_envelope_median(cohorts, path):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.axhspan(70, 180, color="lightgreen", alpha=0.18)
    h = np.arange(24)
    for n in ORDER:
        med = cohorts[n]["diurnal_median"]
        p25 = cohorts[n]["diurnal_p25"]
        p75 = cohorts[n]["diurnal_p75"]
        ax.plot(h, med, color=COL[n], lw=2.2, marker="o", ms=4, label=f"{n} median")
        ax.fill_between(h, p25, p75, color=COL[n], alpha=0.15)
    ax.set_xticks(np.arange(0, 25, 3))
    ax.set_xlabel("hour of day")
    ax.set_ylabel("BG (mg/dL)")
    ax.set_title("Diurnal BG: per-record median and IQR (p25–p75) envelope")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_weekday_heatmaps(cohorts, path):
    fig, axes = plt.subplots(1, len(ORDER), figsize=(4.5 * len(ORDER), 4.5))
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    vmin, vmax = 90, 220
    im = None
    for ax, n in zip(axes, ORDER):
        g = cohorts[n]["wd_grid"]
        im = ax.imshow(g, aspect="auto", origin="upper", cmap="viridis",
                       vmin=vmin, vmax=vmax)
        ax.set_yticks(range(7))
        ax.set_yticklabels(days)
        ax.set_xticks(np.arange(0, 24, 3))
        ax.set_xlabel("hour")
        ax.set_title(n)
    fig.suptitle("Mean BG by weekday × hour", fontweight="bold")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85, label="BG (mg/dL)")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_weekday_heatmaps_median(cohorts, path):
    fig, axes = plt.subplots(1, len(ORDER), figsize=(4.5 * len(ORDER), 4.5))
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    vmin, vmax = 90, 220
    im = None
    for ax, n in zip(axes, ORDER):
        g = cohorts[n]["wd_grid_median"]
        im = ax.imshow(g, aspect="auto", origin="upper", cmap="viridis",
                       vmin=vmin, vmax=vmax)
        ax.set_yticks(range(7))
        ax.set_yticklabels(days)
        ax.set_xticks(np.arange(0, 24, 3))
        ax.set_xlabel("hour")
        ax.set_title(n)
    fig.suptitle("Median BG by weekday × hour", fontweight="bold")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85, label="BG (mg/dL)")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_acf(cohorts, path):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for n in ORDER:
        c = cohorts[n]
        L = c["lags_min"]
        v = [c["pooled_acf"][l] for l in L]
        ax.plot(L, v, color=COL[n], lw=2.0, marker="o", ms=5, label=n)
    ax.set_xscale("log")
    ax.set_xticks([5, 15, 30, 60, 120, 240, 480, 720, 1440])
    ax.set_xticklabels(["5m", "15m", "30m", "1h", "2h", "4h", "8h", "12h", "24h"])
    ax.axhline(0, color="grey", lw=0.7)
    ax.set_ylabel("autocorrelation")
    ax.set_xlabel("lag")
    ax.set_title("Autocorrelation of CGM values across lag")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_delta_distribution(cohorts, path):
    """Each cohort's Δ-BG at its native cadence — annotate clearly that
    cadence differs between Shanghai (15m) and others (5m)."""
    fig, ax = plt.subplots(figsize=(11, 5.5))
    bins = np.arange(-40, 41, 1.0)
    labels = {"Ohio": "Ohio (Δ5m)", "Shanghai": "Shanghai (Δ15m)",
              "AZT1D": "AZT1D (Δ5m)", "Sim": "T1DMSIM (Δ5m)"}
    for n in ORDER:
        d = cohorts[n]["pooled_delta"]
        if len(d) == 0:
            continue
        ax.hist(d, bins=bins, density=True, histtype="step", lw=2.0,
                color=COL[n], label=labels.get(n, n))
    ax.set_yscale("log")
    ax.set_xlabel("ΔBG between consecutive CGM samples (mg/dL)")
    ax.set_ylabel("density (log)")
    ax.set_title("Rate-of-change distribution (native cadence per cohort)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_risk_indices(cohorts, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    jitter = np.random.default_rng(0)  # seeded: the figure must be reproducible
    for ax, key, title in [(axes[0], "LBGI", "LBGI — low BG risk index"),
                           (axes[1], "HBGI", "HBGI — high BG risk index")]:
        data = []
        for n in ORDER:
            vals = [p[key] for p in cohorts[n]["per"]
                    if key in p and not np.isnan(p[key])]
            data.append(vals)
        bp = ax.boxplot(data, tick_labels=ORDER, patch_artist=True, widths=0.55,
                        medianprops=dict(color="black", lw=2))
        for patch, n in zip(bp["boxes"], ORDER):
            patch.set_facecolor(COL[n])
            patch.set_alpha(0.6)
        for i, vals in enumerate(data, start=1):
            for v in vals:
                ax.plot(i + jitter.uniform(-0.07, 0.07), v, ".",
                        color="black", alpha=0.55, ms=4)
        ax.set_title(title, fontweight="bold")
        ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_per_patient_scatter(cohorts, path):
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    for n in ORDER:
        xs = [p.get("TBR1_pct", 0) + p.get("TBR2_pct", 0) for p in cohorts[n]["per"]]
        ys = [p.get("TIR_pct", 0) for p in cohorts[n]["per"]]
        ax.scatter(xs, ys, color=COL[n], alpha=0.7, s=45,
                   edgecolor="black", linewidth=0.4, label=n)
    ax.axhline(70, color="grey", lw=0.7, ls=":")
    ax.axvline(4, color="grey", lw=0.7, ls=":")
    ax.text(4.2, 71, "ADA TIR≥70, TBR≤4", fontsize=8, color="grey")
    ax.set_xlabel("TBR (% time <70 mg/dL)")
    ax.set_ylabel("TIR (% time 70–180 mg/dL)")
    ax.set_title("Per-record scatter — TIR vs total TBR")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_episode_durations(cohorts, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, key, title in [(axes[0], "hypo_durs", "Hypo <70 mg/dL"),
                           (axes[1], "hyper_durs", "Hyper >180 mg/dL")]:
        data = [cohorts[n][key] for n in ORDER]
        bp = ax.boxplot(data, tick_labels=ORDER, patch_artist=True, widths=0.55,
                        medianprops=dict(color="black", lw=2),
                        flierprops=dict(marker=".", alpha=0.4, markersize=3))
        for patch, n in zip(bp["boxes"], ORDER):
            patch.set_facecolor(COL[n])
            patch.set_alpha(0.6)
        for i, d in enumerate(data, start=1):
            ax.text(i, ax.get_ylim()[1] * 0.95, f"n={len(d):,}",
                    ha="center", fontsize=9, color="dimgray")
        ax.set_yscale("log")
        ax.set_ylabel("duration (min)")
        ax.set_title(title, fontweight="bold")
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Episode duration distributions (log y)", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_variability_metrics(cohorts, path):
    metrics = [("cv_pct", "CV %"), ("mage", "MAGE"),
               ("conga_1h", "CONGA-1h"), ("conga_4h", "CONGA-4h"),
               ("modd", "MODD"), ("j_index", "J-index"),
               ("m_value", "M-value"), ("sample_entropy", "SampEn")]
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5))
    for ax, (key, title) in zip(axes.ravel(), metrics):
        data = []
        for n in ORDER:
            data.append([p[key] for p in cohorts[n]["per"]
                         if key in p and not np.isnan(p[key])])
        bp = ax.boxplot(data, tick_labels=ORDER, patch_artist=True, widths=0.55,
                        medianprops=dict(color="black", lw=2))
        for patch, n in zip(bp["boxes"], ORDER):
            patch.set_facecolor(COL[n])
            patch.set_alpha(0.6)
        ax.set_title(title, fontweight="bold")
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Per-record variability / glycemic indices",
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_pct_table_figure(cohorts, path):
    pcts = (1, 5, 10, 25, 50, 75, 90, 95, 99)
    rows = []
    for n in ORDER:
        d = cohorts[n]["pooled_bg"]
        rows.append([np.percentile(d, p) for p in pcts])
    rows = np.array(rows)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for i, n in enumerate(ORDER):
        ax.plot(pcts, rows[i], color=COL[n], lw=2.2, marker="o", ms=5, label=n)
    ax.set_xlabel("percentile of pooled CGM distribution")
    ax.set_ylabel("BG (mg/dL)")
    ax.set_title("Distribution percentile curves")
    ax.axhspan(70, 180, color="lightgreen", alpha=0.18)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_qq(cohorts, path):
    """Quantile-Quantile plots Sim vs each real cohort."""
    refs = [n for n in REAL_ORDER if n in cohorts and len(cohorts[n]["pooled_bg"])]
    fig, axes = plt.subplots(1, len(refs), figsize=(5.5 * len(refs), 5.5))
    if len(refs) == 1:
        axes = [axes]
    qs = np.linspace(0.01, 0.99, 200)
    sim_q = np.quantile(cohorts["Sim"]["pooled_bg"], qs)
    for ax, ref in zip(axes, refs):
        ref_q = np.quantile(cohorts[ref]["pooled_bg"], qs)
        ax.plot(ref_q, sim_q, color=COL["Sim"], lw=2.0, label="data")
        lo = min(ref_q.min(), sim_q.min())
        hi = max(ref_q.max(), sim_q.max())
        ax.plot([lo, hi], [lo, hi], color="grey", lw=0.8, ls="--", label="y=x")
        ax.set_xlabel(f"{ref} quantile (mg/dL)")
        ax.set_ylabel("Sim quantile (mg/dL)")
        ax.set_title(f"Sim vs {ref}")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle("Q-Q comparison of pooled CGM distributions",
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_recovery(cohorts, path):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    data = [cohorts[n]["recov_times"] for n in ORDER]
    bp = ax.boxplot(data, tick_labels=ORDER, patch_artist=True, widths=0.55,
                    medianprops=dict(color="black", lw=2),
                    flierprops=dict(marker=".", alpha=0.4, markersize=3))
    for patch, n in zip(bp["boxes"], ORDER):
        patch.set_facecolor(COL[n])
        patch.set_alpha(0.6)
    for i, d in enumerate(data, start=1):
        if d:
            ax.text(i, ax.get_ylim()[1] * 0.93, f"n={len(d):,}",
                    ha="center", fontsize=9, color="dimgray")
    ax.set_yscale("log")
    ax.set_ylabel("time from BG<70 to BG≥80 (min, log)")
    ax.set_title("Hypo recovery time — from first sub-70 sample to first ≥80",
                 fontweight="bold")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_diurnal_lines(cohorts, path):
    """Clean line-overlay of hour-of-day mean BG across cohorts.

    The envelope plot (`diurnal_envelope.png`) shows ±1σ bands that visually
    overlap and obscure direct shape comparison. This figure plots only the
    three mean curves so the diurnal shape is the focus.
    """
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.axhspan(70, 180, color="lightgreen", alpha=0.18, label="TIR (70-180)")
    h = np.arange(24)
    for n in ORDER:
        m = cohorts[n]["diurnal_mean"]
        ax.plot(h, m, color=COL[n], lw=2.5, marker="o", ms=5, label=n)
    ax.set_xticks(np.arange(0, 25, 2))
    ax.set_xlabel("hour of day")
    ax.set_ylabel("BG (mg/dL)")
    ax.set_title("Diurnal BG curves — direct shape comparison "
                 "(per-record mean, no envelope)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_class_balance(cohorts, path):
    """Stacked bar of TBR2/TBR1/TIR/TAR1/TAR2 % per cohort.

    Per-record mean across each cohort. Companion to the
    `clinical_ranges.png` README chart, but built from the same data as the
    rest of the report and labelled with absolute percentages on each segment.
    """
    band_keys = ["TBR2_pct", "TBR1_pct", "TIR_pct", "TAR1_pct", "TAR2_pct"]
    band_labels = ["TBR2 <54", "TBR1 54-70", "TIR 70-180",
                   "TAR1 180-250", "TAR2 >250"]
    band_colors = ["#7E1416", "#E66767", "#86C893", "#F2C744", "#D87F3E"]

    vals = {n: [] for n in ORDER}
    for n in ORDER:
        for k in band_keys:
            vs = [r[k] for r in cohorts[n]["per"] if k in r]
            vals[n].append(float(np.mean(vs)) if vs else 0.0)

    fig, ax = plt.subplots(figsize=(10, 5.2))
    x = np.arange(len(ORDER))
    bottom = np.zeros(len(ORDER))
    for i, (label, color) in enumerate(zip(band_labels, band_colors)):
        heights = np.array([vals[n][i] for n in ORDER])
        ax.bar(x, heights, bottom=bottom, label=label, color=color,
               edgecolor="white", linewidth=1.2)
        for j, v in enumerate(heights):
            if v >= 2.5:
                ax.text(x[j], bottom[j] + v / 2, f"{v:.1f}",
                        ha="center", va="center", fontsize=10,
                        color="black", fontweight="bold")
        bottom += heights
    ax.set_xticks(x)
    ax.set_xticklabels(ORDER)
    ax.set_ylabel("% of samples (per-record mean across cohort)")
    ax.set_ylim(0, 102)
    ax.set_title("BG class distribution — sample-level class balance per cohort",
                 fontweight="bold")
    ax.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left",
              fontsize=9, title="band")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Markdown report writer
# ============================================================================
def _sci(p, sig=1):
    """Format a (small) p-value like '3.5 × 10⁻⁴⁶' or '< 10⁻³⁰⁰'."""
    if not np.isfinite(p) or p <= 0:
        return "< 10⁻³⁰⁰"
    e = int(np.floor(np.log10(p)))
    m = p / (10 ** e)
    if e <= -300:
        return "< 10⁻³⁰⁰"
    sup = str(e).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"{m:.{sig}f} × 10{sup}"


def _ms(summary, key, fmt=".2f"):
    """summary[key] -> 'mean ± std' (returns '—' if key missing)."""
    if key not in summary:
        return "—"
    s = summary[key]
    return f"{s['mean']:{fmt}} ± {s['std']:{fmt}}"


def _ms1(summary, key):
    return _ms(summary, key, ".1f")


def _delta(a, b, fmt="+.1f"):
    return f"{a - b:{fmt}}"


def _acf_threshold_lag(acf_dict, threshold):
    """Lag (in minutes) where ACF first drops below threshold (linear interp).

    `acf_dict` is the per-cohort `pooled_acf` (int-lag → mean ACF). Returns
    +inf if the curve never crosses, or the smallest sampled lag if already
    below at the first sample.
    """
    pairs = sorted((int(k), v) for k, v in acf_dict.items()
                   if v is not None and not (isinstance(v, float) and np.isnan(v)))
    if not pairs:
        return float("nan")
    if pairs[0][1] < threshold:
        return float(pairs[0][0])
    for i in range(len(pairs) - 1):
        l1, v1 = pairs[i]
        l2, v2 = pairs[i + 1]
        if v1 >= threshold and v2 < threshold:
            frac = (v1 - threshold) / (v1 - v2) if v1 != v2 else 0.0
            return l1 + frac * (l2 - l1)
    return float("inf")


def _build_ml_section(cohorts, distances, pooled_moments, pooled_percentiles,
                     cohort_summaries):
    """Compose the Section 0 markdown — ML-relevant simulator statistics.

    Numbers come from the same per-cohort structures used elsewhere in the
    report, so this section stays in sync on every re-run.
    """
    O, S, A, M = "Ohio", "Shanghai", "AZT1D", "Sim"
    pm = pooled_moments
    pp = pooled_percentiles

    def _samples(n):
        return int(pm[n]["n"])

    def _days(n):
        return sum(p["days"] for p in cohorts[n]["per"])

    def _records(n):
        return len(cohorts[n]["per"])

    hours = {n: _days(n) * 24 for n in ORDER}

    def _class_pct(n, key):
        vs = [r[key] for r in cohorts[n]["per"] if key in r]
        return float(np.mean(vs)) if vs else float("nan")

    n_hypo = {n: len(cohorts[n]["hypo_durs"]) for n in ORDER}
    n_hyper = {n: len(cohorts[n]["hyper_durs"]) for n in ORDER}
    n_severe_hypo = {n: len(cohorts[n]["severe_hypo_durs"]) for n in ORDER}
    n_severe_hyper = {n: len(cohorts[n]["severe_hyper_durs"]) for n in ORDER}

    decorr_05 = {n: _acf_threshold_lag(cohorts[n]["pooled_acf"], 0.5)
                 for n in ORDER}
    decorr_02 = {n: _acf_threshold_lag(cohorts[n]["pooled_acf"], 0.2)
                 for n in ORDER}

    between_std = {n: float(np.std([r["mean"] for r in cohorts[n]["per"]]))
                   for n in ORDER}
    within_std = {n: float(np.mean([r["std"] for r in cohorts[n]["per"]]))
                  for n in ORDER}

    d_so = distances["Sim_vs_Ohio"]
    d_ss = distances["Sim_vs_Shanghai"]
    d_sa = distances["Sim_vs_AZT1D"]
    d_oa = distances["Ohio_vs_AZT1D"]
    d_oo = distances["Ohio_vs_Shanghai"]

    def _fmt_lag(min_val):
        if not np.isfinite(min_val):
            return ">24 h"
        h = min_val / 60.0
        return f"{h:.1f} h" if h >= 1.0 else f"{int(round(min_val))} min"

    def _fmt_int(n):
        return f"{n:,}"

    md = f"""## 0. Machine-learning summary

### 0.1 Data volume per cohort

| Cohort | Records | Samples | Hours | CGM-days | Cadence |
|---|---:|---:|---:|---:|---:|
| OhioT1DM     | {_records(O):>3} | {_fmt_int(_samples(O))} | {hours[O]:>7,.0f} | {_days(O):>7,.0f} | {cohorts[O]['step_min']} min |
| ShanghaiT1DM | {_records(S):>3} | {_fmt_int(_samples(S))} | {hours[S]:>7,.0f} | {_days(S):>7,.0f} | {cohorts[S]['step_min']} min |
| AZT1D        | {_records(A):>3} | {_fmt_int(_samples(A))} | {hours[A]:>7,.0f} | {_days(A):>7,.0f} | {cohorts[A]['step_min']} min |
| **T1DMSIM** *(this run)* | **{_records(M)}** | **{_fmt_int(_samples(M))}** | **{hours[M]:>7,.0f}** | **{_days(M):>7,.0f}** | **{cohorts[M]['step_min']} min** |

### 0.2 Normalization statistics

| Stat (mg/dL) | Ohio | Shanghai | AZT1D | **Sim** |
|---|---:|---:|---:|---:|
| mean    | {pm[O]['mean']:.1f}  | {pm[S]['mean']:.1f}  | {pm[A]['mean']:.1f}  | **{pm[M]['mean']:.1f}**  |
| median  | {pm[O]['median']:.1f}  | {pm[S]['median']:.1f}  | {pm[A]['median']:.1f}  | **{pm[M]['median']:.1f}**  |
| std     | {pm[O]['std']:.1f}   | {pm[S]['std']:.1f}   | {pm[A]['std']:.1f}   | **{pm[M]['std']:.1f}**   |
| IQR     | {pm[O]['iqr']:.1f}   | {pm[S]['iqr']:.1f}   | {pm[A]['iqr']:.1f}   | **{pm[M]['iqr']:.1f}**   |
| p1      | {pp[O]['p1']:.1f}    | {pp[S]['p1']:.1f}    | {pp[A]['p1']:.1f}    | **{pp[M]['p1']:.1f}**    |
| p99     | {pp[O]['p99']:.1f}   | {pp[S]['p99']:.1f}   | {pp[A]['p99']:.1f}   | **{pp[M]['p99']:.1f}**   |
| min     | {pm[O]['min']:.1f}   | {pm[S]['min']:.1f}   | {pm[A]['min']:.1f}   | **{pm[M]['min']:.1f}**   |
| max     | {pm[O]['max']:.1f}   | {pm[S]['max']:.1f}   | {pm[A]['max']:.1f}   | **{pm[M]['max']:.1f}**   |

### 0.3 Sample-level class balance

![Class balance per cohort](figures/class_balance.png)

| Band | Threshold | Ohio % | Shang % | AZT1D % | **Sim %** |
|---|---|---:|---:|---:|---:|
| TBR2 | <54     | {_class_pct(O,'TBR2_pct'):>5.2f} | {_class_pct(S,'TBR2_pct'):>5.2f} | {_class_pct(A,'TBR2_pct'):>5.2f} | **{_class_pct(M,'TBR2_pct'):>5.2f}** |
| TBR1 | 54-70   | {_class_pct(O,'TBR1_pct'):>5.2f} | {_class_pct(S,'TBR1_pct'):>5.2f} | {_class_pct(A,'TBR1_pct'):>5.2f} | **{_class_pct(M,'TBR1_pct'):>5.2f}** |
| TIR  | 70-180  | {_class_pct(O,'TIR_pct'):>5.1f}  | {_class_pct(S,'TIR_pct'):>5.1f}  | {_class_pct(A,'TIR_pct'):>5.1f}  | **{_class_pct(M,'TIR_pct'):>5.1f}**  |
| TAR1 | 180-250 | {_class_pct(O,'TAR1_pct'):>5.1f} | {_class_pct(S,'TAR1_pct'):>5.1f} | {_class_pct(A,'TAR1_pct'):>5.1f} | **{_class_pct(M,'TAR1_pct'):>5.1f}** |
| TAR2 | >250    | {_class_pct(O,'TAR2_pct'):>5.1f} | {_class_pct(S,'TAR2_pct'):>5.1f} | {_class_pct(A,'TAR2_pct'):>5.1f} | **{_class_pct(M,'TAR2_pct'):>5.1f}** |

### 0.4 Episode-level event counts

| Event class | Ohio | Shanghai | AZT1D | **Sim** |
|---|---:|---:|---:|---:|
| Hypo (<70) episodes        | {_fmt_int(n_hypo[O])}   | {_fmt_int(n_hypo[S])}   | {_fmt_int(n_hypo[A])}   | **{_fmt_int(n_hypo[M])}** |
| Severe-hypo (<54) episodes | {_fmt_int(n_severe_hypo[O])}    | {_fmt_int(n_severe_hypo[S])}    | {_fmt_int(n_severe_hypo[A])}    | **{_fmt_int(n_severe_hypo[M])}** |
| Hyper (>180) episodes      | {_fmt_int(n_hyper[O])}  | {_fmt_int(n_hyper[S])}  | {_fmt_int(n_hyper[A])}  | **{_fmt_int(n_hyper[M])}** |
| Severe-hyper (>250) episodes | {_fmt_int(n_severe_hyper[O])}  | {_fmt_int(n_severe_hyper[S])}  | {_fmt_int(n_severe_hyper[A])}  | **{_fmt_int(n_severe_hyper[M])}** |

### 0.5 Effective context window

| ACF threshold | Ohio | Shanghai | AZT1D | **Sim** |
|---|---:|---:|---:|---:|
| 0.5 (50% retained) | {_fmt_lag(decorr_05[O])} | {_fmt_lag(decorr_05[S])} | {_fmt_lag(decorr_05[A])} | **{_fmt_lag(decorr_05[M])}** |
| 0.2 (20% retained) | {_fmt_lag(decorr_02[O])} | {_fmt_lag(decorr_02[S])} | {_fmt_lag(decorr_02[A])} | **{_fmt_lag(decorr_02[M])}** |

### 0.6 Cross-record heterogeneity

| Cohort | Between-patient mean-BG std | Within-patient BG std | Ratio |
|---|---:|---:|---:|
| Ohio     | {between_std[O]:.1f} | {within_std[O]:.1f} | {between_std[O]/max(1e-6,within_std[O]):.2f} |
| Shanghai | {between_std[S]:.1f} | {within_std[S]:.1f} | {between_std[S]/max(1e-6,within_std[S]):.2f} |
| AZT1D    | {between_std[A]:.1f} | {within_std[A]:.1f} | {between_std[A]/max(1e-6,within_std[A]):.2f} |
| **Sim**  | **{between_std[M]:.1f}** | **{within_std[M]:.1f}** | **{between_std[M]/max(1e-6,within_std[M]):.2f}** |

### 0.7 Diurnal shape (clean line overlay)

![Diurnal BG curves — clean line overlay](figures/diurnal_lines.png)

### 0.8 Sim-vs-real domain gap

| Pair | KS | Wasserstein-1 (mg/dL) | JS divergence |
|---|---:|---:|---:|
| **Sim vs Ohio**     | {d_so['ks_stat']:.3f} | {d_so['wasserstein']:.1f} | {d_so['js_div']:.3f} |
| **Sim vs Shanghai** | {d_ss['ks_stat']:.3f} | {d_ss['wasserstein']:.1f} | {d_ss['js_div']:.3f} |
| **Sim vs AZT1D**    | {d_sa['ks_stat']:.3f} | {d_sa['wasserstein']:.1f} | {d_sa['js_div']:.3f} |
| Ohio vs Shanghai (real-vs-real baseline) | {d_oo['ks_stat']:.3f} | {d_oo['wasserstein']:.1f} | {d_oo['js_div']:.3f} |
| Ohio vs AZT1D    (real-vs-real baseline) | {d_oa['ks_stat']:.3f} | {d_oa['wasserstein']:.1f} | {d_oa['js_div']:.3f} |
| Shanghai vs AZT1D (real-vs-real baseline) | {distances['Shanghai_vs_AZT1D']['ks_stat']:.3f} | {distances['Shanghai_vs_AZT1D']['wasserstein']:.1f} | {distances['Shanghai_vs_AZT1D']['js_div']:.3f} |
"""
    return md


def write_report_md(cohorts, distances, pooled_moments, pooled_percentiles,
                    pooled_risk, cohort_summaries, recov_summaries,
                    az_event_stats, sim_event_stats, unexpl_stats, ext, path):
    """Template the full markdown report from computed stats.

    Tables are filled programmatically from the same numbers that go into
    stats.json. Prose is kept neutral and observational (raw deltas, no
    "matches"/"diverges" verdicts) so re-runs after simulator changes do not
    require hand-editing the report.
    """
    n = {x: cohorts[x] for x in ORDER}
    pm = pooled_moments
    pp = pooled_percentiles
    pr = pooled_risk
    sm = cohort_summaries
    rec = recov_summaries

    # Convenience handles
    O, S, A, M = "Ohio", "Shanghai", "AZT1D", "Sim"

    # Per-record TIR IQR + mean-BG std across records (Section 8)
    def _per_records_field(cohort_name, key):
        return [r[key] for r in cohorts[cohort_name]["per"]
                if key in r and not np.isnan(r[key])]

    tir_iqr = {x: float(np.percentile(_per_records_field(x, "TIR_pct"), 75)
                        - np.percentile(_per_records_field(x, "TIR_pct"), 25))
               for x in ORDER}
    tir_lo = {x: float(np.min(_per_records_field(x, "TIR_pct"))) for x in ORDER}
    tir_hi = {x: float(np.max(_per_records_field(x, "TIR_pct"))) for x in ORDER}
    mean_bg_std = {x: float(np.std(_per_records_field(x, "mean"))) for x in ORDER}

    # ACF lag rows that exist at 5-min cadence (Ohio, AZT1D, Sim) and at
    # 15-min cadence (Shanghai). The 5-min row is absent for Shanghai.
    acf = {x: cohorts[x]["pooled_acf"] for x in ORDER}

    def _acf_cell(name, lag_min):
        v = acf[name].get(lag_min, acf[name].get(str(lag_min), float("nan")))
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "(n/a)"
        return f"{v:.3f}"

    # Δ vs each real cohort
    def pct_row(pkey):
        po = pp[O][pkey]; ps = pp[S][pkey]; pa = pp[A][pkey]; psim = pp[M][pkey]
        return (f"| {pkey} | {po:.1f} | {ps:.1f} | {pa:.1f} | {psim:.1f} | "
                f"{psim-po:+.1f} | {psim-ps:+.1f} | {psim-pa:+.1f} |")

    pct_rows = "\n".join(pct_row(f"p{p}") for p in (1, 5, 10, 25, 50, 75, 90, 95, 99))

    # Diurnal hourly means / medians
    dm = {x: cohorts[x]["diurnal_mean"] for x in ORDER}
    dmed = {x: cohorts[x]["diurnal_median"] for x in ORDER}

    def hour_row(name):
        return " | ".join(f"{dm[name][h]:.0f}" for h in range(24))

    def hour_row_median(name):
        return " | ".join(f"{dmed[name][h]:.0f}" for h in range(24))

    # Distances table — every pair we computed.
    dist_rows = []
    pair_labels = [
        ("Ohio_vs_Shanghai", "Ohio vs Shanghai"),
        ("Ohio_vs_AZT1D", "Ohio vs AZT1D"),
        ("Shanghai_vs_AZT1D", "Shanghai vs AZT1D"),
        ("Sim_vs_Ohio", "Sim vs Ohio"),
        ("Sim_vs_Shanghai", "Sim vs Shanghai"),
        ("Sim_vs_AZT1D", "Sim vs AZT1D"),
    ]
    for pair_key, label in pair_labels:
        if pair_key not in distances:
            continue
        d = distances[pair_key]
        dist_rows.append(
            f"| {label} | {d['ks_stat']:.3f} | {_sci(d['ks_p'])} | "
            f"{d['wasserstein']:.1f} | {d['js_div']:.3f} |"
        )
    dist_table = "\n".join(dist_rows)

    # Recovery table
    def rec_row(name):
        r = rec[name]
        return (f"| {name:8} | {r['n']:>5,} | {r['median']:.0f} | "
                f"{r['p75']:.0f} | {r['p90']:.0f} | {r['p99']:.0f} | {r['max']:.0f} |")

    rec_table = "\n".join(rec_row(x) for x in ORDER)

    # §6.3 unexplained-excursion table
    ueO, ueA, ueM = unexpl_stats["Ohio"], unexpl_stats["AZT1D"], unexpl_stats["Sim"]

    def _uerow(label, key, fmt="{:.1f}"):
        return (f"| {label} | {fmt.format(ueO[key])} | "
                f"{fmt.format(ueA[key])} | {fmt.format(ueM[key])} |")

    unexpl_section = f"""![Unexplained-excursion summary](figures/unexplained_summary.png)

![Unexplained excursions in real CGM](figures/unexplained_gallery.png)

| Quantity | OhioT1DM | AZT1D | T1DMSIM |
|---|---:|---:|---:|
{_uerow("Excursions detected", "n_exc", "{:.0f}")}
{_uerow("&nbsp;&nbsp;rises unexplained (%)", "rise_unexpl_pct")}
{_uerow("&nbsp;&nbsp;falls unexplained (%)", "fall_unexpl_pct")}
{_uerow("&nbsp;&nbsp;all unexplained (%)", "all_unexpl_pct")}
{_uerow("Explained load (mg/dL/day)", "load_expl", "{:.0f}")}
{_uerow("Unexplained load (mg/dL/day)", "load_unexpl", "{:.0f}")}
{_uerow("Median amplitude, explained (mg/dL)", "amp_expl", "{:.0f}")}
{_uerow("Median amplitude, unexplained (mg/dL)", "amp_unexpl", "{:.0f}")}
{_uerow("Δ-BG SD, full trace (mg/dL)", "dbg_sd_full", "{:.2f}")}
{_uerow("Δ-BG SD, unexplained censored (mg/dL)", "dbg_sd_cens", "{:.2f}")}"""

    # Synthesis tables — produce neutral side-by-side rows with Δ columns.
    def syn_row(label, sim_val, ohio_val, shang_val, az_val, fmt=".1f", unit=""):
        return (f"| {label} | {sim_val:{fmt}}{unit} | {ohio_val:{fmt}}{unit} | "
                f"{shang_val:{fmt}}{unit} | {az_val:{fmt}}{unit} | "
                f"{sim_val-ohio_val:+{fmt}} | {sim_val-shang_val:+{fmt}} | "
                f"{sim_val-az_val:+{fmt}} |")

    # ML-friendly section (volume, normalization, class balance, autocorr, ...)
    ext_section_md = _build_extended_section(ext, cohorts)
    ml_section_md = _build_ml_section(cohorts, distances, pooled_moments,
                                       pooled_percentiles, cohort_summaries)

    # AZT1D event section
    az_pool = az_event_stats["pooled"]
    sim_pool = sim_event_stats["pooled"]

    md = f"""# T1DMSIM vs OhioT1DM, ShanghaiT1DM, AZT1D — Statistical Comparison Report

{ml_section_md}

## 1. Corpora at a glance

| Dataset | Records | Cadence | Total CGM-days | Cohort | Notes |
|---|---:|---:|---:|---|---|
| OhioT1DM | {len(n[O]['per'])} records (file pairs) | {n[O]['step_min']} min Dexcom | {sum(p['days'] for p in n[O]['per']):.1f} | US adults, pump + announced meals | training + testing periods concatenated per patient |
| ShanghaiT1DM | {len(n[S]['per'])} records | **{n[S]['step_min']} min** | {sum(p['days'] for p in n[S]['per']):.1f} | CN adults, mixed CSII + MDI (incl. regular Novolin R), BMI ≈ 21 | shorter individual records (~10 d) |
| AZT1D | {len(n[A]['per'])} subjects | {n[A]['step_min']} min Dexcom G6 | {sum(p['days'] for p in n[A]['per']):.1f} | US adults, Mayo Clinic AZ, all on Tandem t:slim X2 Control-IQ (AID) | rich pump event log: bolus type, basal rate, carbs, device mode |
| T1DMSIM *(this run)* | {len(n[M]['per'])} seeds × {int(round(np.mean([p['days'] for p in n[M]['per']])))} days | {n[M]['step_min']} min | {sum(p['days'] for p in n[M]['per']):.1f} | synthetic, seeds 0–{len(n[M]['per'])-1}, 24 h warm-up discarded | `initial_bg = 120 mg/dL`, `bg_observed` (sensor-noised); generator is unbounded |

## 2. Headline numbers

### 2.1 Pooled central moments

| Metric (mg/dL) | OhioT1DM | ShanghaiT1DM | AZT1D | T1DMSIM | Sim − Ohio | Sim − Shang | Sim − AZT1D |
|---|---:|---:|---:|---:|---:|---:|---:|
| n (samples) | {pm[O]['n']:,} | {pm[S]['n']:,} | {pm[A]['n']:,} | {pm[M]['n']:,} | — | — | — |
| **mean** | {pm[O]['mean']:.1f} | {pm[S]['mean']:.1f} | {pm[A]['mean']:.1f} | {pm[M]['mean']:.1f} | {pm[M]['mean']-pm[O]['mean']:+.1f} | {pm[M]['mean']-pm[S]['mean']:+.1f} | {pm[M]['mean']-pm[A]['mean']:+.1f} |
| **median** | {pm[O]['median']:.1f} | {pm[S]['median']:.1f} | {pm[A]['median']:.1f} | {pm[M]['median']:.1f} | {pm[M]['median']-pm[O]['median']:+.1f} | {pm[M]['median']-pm[S]['median']:+.1f} | {pm[M]['median']-pm[A]['median']:+.1f} |
| std | {pm[O]['std']:.1f} | {pm[S]['std']:.1f} | {pm[A]['std']:.1f} | {pm[M]['std']:.1f} | {pm[M]['std']-pm[O]['std']:+.1f} | {pm[M]['std']-pm[S]['std']:+.1f} | {pm[M]['std']-pm[A]['std']:+.1f} |
| IQR | {pm[O]['iqr']:.1f} | {pm[S]['iqr']:.1f} | {pm[A]['iqr']:.1f} | {pm[M]['iqr']:.1f} | {pm[M]['iqr']-pm[O]['iqr']:+.1f} | {pm[M]['iqr']-pm[S]['iqr']:+.1f} | {pm[M]['iqr']-pm[A]['iqr']:+.1f} |
| CV (%) | {pm[O]['cv_pct']:.1f} | {pm[S]['cv_pct']:.1f} | {pm[A]['cv_pct']:.1f} | {pm[M]['cv_pct']:.1f} | {pm[M]['cv_pct']-pm[O]['cv_pct']:+.1f} pp | {pm[M]['cv_pct']-pm[S]['cv_pct']:+.1f} pp | {pm[M]['cv_pct']-pm[A]['cv_pct']:+.1f} pp |
| skewness | {pm[O]['skew']:.2f} | {pm[S]['skew']:.2f} | {pm[A]['skew']:.2f} | {pm[M]['skew']:.2f} | {pm[M]['skew']-pm[O]['skew']:+.2f} | {pm[M]['skew']-pm[S]['skew']:+.2f} | {pm[M]['skew']-pm[A]['skew']:+.2f} |
| excess kurtosis | {pm[O]['excess_kurt']:.2f} | {pm[S]['excess_kurt']:.2f} | {pm[A]['excess_kurt']:.2f} | {pm[M]['excess_kurt']:.2f} | {pm[M]['excess_kurt']-pm[O]['excess_kurt']:+.2f} | {pm[M]['excess_kurt']-pm[S]['excess_kurt']:+.2f} | {pm[M]['excess_kurt']-pm[A]['excess_kurt']:+.2f} |
| min | {pm[O]['min']:.1f} | {pm[S]['min']:.1f} | {pm[A]['min']:.1f} | {pm[M]['min']:.1f} | — | — | — |
| max | {pm[O]['max']:.1f} | {pm[S]['max']:.1f} | {pm[A]['max']:.1f} | {pm[M]['max']:.1f} | — | — | — |

### 2.2 Percentiles of the pooled distribution

| Percentile | OhioT1DM | ShanghaiT1DM | AZT1D | T1DMSIM | Sim − Ohio | Sim − Shang | Sim − AZT1D |
|---|---:|---:|---:|---:|---:|---:|---:|
{pct_rows}

![Percentile curves](figures/percentile_curves.png)

![Pooled PDF](figures/pdf_pooled.png)

![Pooled empirical CDF](figures/cdf_pooled.png)

![Q-Q vs each real cohort](figures/qq.png)

### 2.3 Distribution-distance statistics

| Pair | KS statistic | KS p-value | Wasserstein-1 (mg/dL) | JS divergence (5 mg/dL bins) |
|---|---:|---:|---:|---:|
{dist_table}

## 3. Clinical glycemic indices

| Index | OhioT1DM | ShanghaiT1DM | AZT1D | T1DMSIM |
|---|---|---|---|---|
| GMI / eA1c proxy | {_ms(sm[O], 'GMI')} | {_ms(sm[S], 'GMI')} | {_ms(sm[A], 'GMI')} | {_ms(sm[M], 'GMI')} |
| **LBGI** (low-BG risk) | {_ms(sm[O], 'LBGI')} | {_ms(sm[S], 'LBGI')} | {_ms(sm[A], 'LBGI')} | **{_ms(sm[M], 'LBGI')}** |
| **HBGI** (high-BG risk) | {_ms(sm[O], 'HBGI')} | {_ms(sm[S], 'HBGI')} | {_ms(sm[A], 'HBGI')} | **{_ms(sm[M], 'HBGI')}** |
| J-index | {_ms1(sm[O], 'j_index')} | {_ms1(sm[S], 'j_index')} | {_ms1(sm[A], 'j_index')} | {_ms1(sm[M], 'j_index')} |
| M-value (ref 120) | {_ms1(sm[O], 'm_value')} | {_ms1(sm[S], 'm_value')} | {_ms1(sm[A], 'm_value')} | {_ms1(sm[M], 'm_value')} |

| | Ohio | Shanghai | AZT1D | Sim |
|---|---:|---:|---:|---:|
| LBGI (pooled) | {pr[O]['LBGI_pooled']:.2f} | {pr[S]['LBGI_pooled']:.2f} | {pr[A]['LBGI_pooled']:.2f} | {pr[M]['LBGI_pooled']:.2f} |
| HBGI (pooled) | {pr[O]['HBGI_pooled']:.2f} | {pr[S]['HBGI_pooled']:.2f} | {pr[A]['HBGI_pooled']:.2f} | {pr[M]['HBGI_pooled']:.2f} |
| J-index (pooled) | {pr[O]['J_index_pooled']:.1f} | {pr[S]['J_index_pooled']:.1f} | {pr[A]['J_index_pooled']:.1f} | {pr[M]['J_index_pooled']:.1f} |
| M-value (pooled) | {pr[O]['M_value_pooled']:.1f} | {pr[S]['M_value_pooled']:.1f} | {pr[A]['M_value_pooled']:.1f} | {pr[M]['M_value_pooled']:.1f} |

### 3.1 Time-in-range, per-record cohort summary

| Range | OhioT1DM | ShanghaiT1DM | AZT1D | T1DMSIM |
|---|---|---|---|---|
| TBR2 (<54)        | {_ms(sm[O], 'TBR2_pct')} | {_ms(sm[S], 'TBR2_pct')} | {_ms(sm[A], 'TBR2_pct')} | {_ms(sm[M], 'TBR2_pct')} |
| TBR1 (54–70)      | {_ms(sm[O], 'TBR1_pct')} | {_ms(sm[S], 'TBR1_pct')} | {_ms(sm[A], 'TBR1_pct')} | {_ms(sm[M], 'TBR1_pct')} |
| **TIR (70–180)**  | **{_ms1(sm[O], 'TIR_pct')}** | **{_ms1(sm[S], 'TIR_pct')}** | **{_ms1(sm[A], 'TIR_pct')}** | **{_ms1(sm[M], 'TIR_pct')}** |
| TAR1 (180–250)    | {_ms1(sm[O], 'TAR1_pct')} | {_ms1(sm[S], 'TAR1_pct')} | {_ms1(sm[A], 'TAR1_pct')} | {_ms1(sm[M], 'TAR1_pct')} |
| TAR2 (>250)       | {_ms(sm[O], 'TAR2_pct')} | {_ms(sm[S], 'TAR2_pct')} | {_ms(sm[A], 'TAR2_pct')} | {_ms(sm[M], 'TAR2_pct')} |

## 4. Variability and complexity

| Metric (native cadence) | OhioT1DM | ShanghaiT1DM | AZT1D | T1DMSIM |
|---|---|---|---|---|
| CV (%)              | {_ms1(sm[O], 'cv_pct')}   | {_ms1(sm[S], 'cv_pct')}   | {_ms1(sm[A], 'cv_pct')}   | **{_ms1(sm[M], 'cv_pct')}**   |
| MAGE (mg/dL)        | {_ms1(sm[O], 'mage')}     | {_ms1(sm[S], 'mage')}¹     | {_ms1(sm[A], 'mage')}     | {_ms1(sm[M], 'mage')}     |
| CONGA-1h (mg/dL)    | {_ms1(sm[O], 'conga_1h')} | {_ms1(sm[S], 'conga_1h')} | {_ms1(sm[A], 'conga_1h')} | {_ms1(sm[M], 'conga_1h')} |
| CONGA-4h (mg/dL)    | {_ms1(sm[O], 'conga_4h')} | {_ms1(sm[S], 'conga_4h')} | {_ms1(sm[A], 'conga_4h')} | {_ms1(sm[M], 'conga_4h')} |
| MODD (mg/dL)        | {_ms1(sm[O], 'modd')}     | {_ms1(sm[S], 'modd')}     | {_ms1(sm[A], 'modd')}     | **{_ms1(sm[M], 'modd')}**     |
| Sample entropy      | {_ms(sm[O], 'sample_entropy')} | {_ms(sm[S], 'sample_entropy')} | {_ms(sm[A], 'sample_entropy')} | {_ms(sm[M], 'sample_entropy')} |

![Variability and complexity panel](figures/variability_metrics.png)

## 5. Temporal dynamics

### 5.1 Autocorrelation

| Lag         | OhioT1DM | ShanghaiT1DM | AZT1D | T1DMSIM |
|---|---|---|---|---|
| 5 min   | {_acf_cell(O, 5)} | {_acf_cell(S, 5)}  | {_acf_cell(A, 5)} | {_acf_cell(M, 5)} |
| 15 min  | {_acf_cell(O, 15)} | {_acf_cell(S, 15)}  | {_acf_cell(A, 15)} | {_acf_cell(M, 15)} |
| 30 min  | {_acf_cell(O, 30)} | {_acf_cell(S, 30)}  | {_acf_cell(A, 30)} | {_acf_cell(M, 30)} |
| 1 h     | {_acf_cell(O, 60)} | {_acf_cell(S, 60)}  | {_acf_cell(A, 60)} | {_acf_cell(M, 60)} |
| 2 h     | {_acf_cell(O, 120)} | {_acf_cell(S, 120)}  | {_acf_cell(A, 120)} | {_acf_cell(M, 120)} |
| 4 h     | {_acf_cell(O, 240)} | {_acf_cell(S, 240)}  | {_acf_cell(A, 240)} | {_acf_cell(M, 240)} |
| **8 h**     | **{_acf_cell(O, 480)}** | **{_acf_cell(S, 480)}** | **{_acf_cell(A, 480)}** | **{_acf_cell(M, 480)}** |
| **12 h**    | **{_acf_cell(O, 720)}** | **{_acf_cell(S, 720)}** | **{_acf_cell(A, 720)}** | **{_acf_cell(M, 720)}** |
| 24 h    | {_acf_cell(O, 1440)} | {_acf_cell(S, 1440)}  | {_acf_cell(A, 1440)} | **{_acf_cell(M, 1440)}** |

![Autocorrelation across lag](figures/acf.png)

### 5.2 Rate-of-change (Δ-BG)

![Δ-BG distribution at native cadence](figures/delta_distribution.png)

### 5.3 Diurnal pattern (hour-of-day across records)

![Hour-of-day mean with ±1σ envelope](figures/diurnal_envelope.png)

![Hour-of-day median with IQR envelope](figures/diurnal_envelope_median.png)

|   | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Ohio | {hour_row(O)} |
| Shanghai | {hour_row(S)} |
| AZT1D | {hour_row(A)} |
| Sim | {hour_row(M)} |

|   | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Ohio | {hour_row_median(O)} |
| Shanghai | {hour_row_median(S)} |
| AZT1D | {hour_row_median(A)} |
| Sim | {hour_row_median(M)} |

![Weekday × hour mean heatmap](figures/weekday_heatmap.png)

![Weekday × hour median heatmap](figures/weekday_heatmap_median.png)

## 6. Excursion-level dynamics

### 6.1 Episode counts and durations

| Metric | OhioT1DM | ShanghaiT1DM | AZT1D | T1DMSIM |
|---|---|---|---|---|
| Hypo (<70) episodes / day      | {_ms(sm[O], 'hypo_count_per_day')} | {_ms(sm[S], 'hypo_count_per_day')} | {_ms(sm[A], 'hypo_count_per_day')} | **{_ms(sm[M], 'hypo_count_per_day')}** |
| Severe-hypo (<54) eps / day   | {_ms(sm[O], 'severe_hypo_count_per_day')} | {_ms(sm[S], 'severe_hypo_count_per_day')} | {_ms(sm[A], 'severe_hypo_count_per_day')} | {_ms(sm[M], 'severe_hypo_count_per_day')} |
| Hyper (>180) episodes / day   | {_ms(sm[O], 'hyper_count_per_day')} | {_ms(sm[S], 'hyper_count_per_day')} | {_ms(sm[A], 'hyper_count_per_day')} | {_ms(sm[M], 'hyper_count_per_day')} |
| Severe-hyper (>250) eps / day | {_ms(sm[O], 'severe_hyper_count_per_day')} | {_ms(sm[S], 'severe_hyper_count_per_day')} | {_ms(sm[A], 'severe_hyper_count_per_day')} | {_ms(sm[M], 'severe_hyper_count_per_day')} |
| Hypo median duration (min)    | {sm[O]['hypo_median_min']['mean']:.1f} | {sm[S]['hypo_median_min']['mean']:.1f} | {sm[A]['hypo_median_min']['mean']:.1f} | {sm[M]['hypo_median_min']['mean']:.1f} |
| Hypo p90 duration (min)       | {sm[O]['hypo_p90_min']['mean']:.1f} | {sm[S]['hypo_p90_min']['mean']:.1f} | {sm[A]['hypo_p90_min']['mean']:.1f} | **{sm[M]['hypo_p90_min']['mean']:.1f}** |
| Hyper median duration (min)   | {sm[O]['hyper_median_min']['mean']:.1f} | {sm[S]['hyper_median_min']['mean']:.1f} | {sm[A]['hyper_median_min']['mean']:.1f} | {sm[M]['hyper_median_min']['mean']:.1f} |
| Hyper p90 duration (min)      | {sm[O]['hyper_p90_min']['mean']:.1f} | {sm[S]['hyper_p90_min']['mean']:.1f} | {sm[A]['hyper_p90_min']['mean']:.1f} | **{sm[M]['hyper_p90_min']['mean']:.1f}** |

![Episode duration boxplots](figures/episode_durations.png)

### 6.2 Hypo recovery time

![Hypo recovery time from BG<70 to BG≥80](figures/recovery_time.png)

| Cohort | n events | median (min) | p75 (min) | p90 (min) | p99 (min) | max (min) |
|---|---:|---:|---:|---:|---:|---:|
{rec_table}

### 6.3 Unexplained excursions

{unexpl_section}

## 7. Per-record (per-patient) heterogeneity

![Per-record TIR vs TBR scatter](figures/per_patient_scatter.png)

| Cohort | TIR IQR (pp) | TIR min–max (pp) | Mean-BG std across records |
|---|---:|---|---:|
| Ohio     | {tir_iqr[O]:.1f} | {tir_lo[O]:.1f} – {tir_hi[O]:.1f} | {mean_bg_std[O]:.1f} |
| Shanghai | {tir_iqr[S]:.1f} | {tir_lo[S]:.1f} – {tir_hi[S]:.1f} | {mean_bg_std[S]:.1f} |
| AZT1D    | {tir_iqr[A]:.1f} | {tir_lo[A]:.1f} – {tir_hi[A]:.1f} | {mean_bg_std[A]:.1f} |
| Sim      | {tir_iqr[M]:.1f} | {tir_lo[M]:.1f} – {tir_hi[M]:.1f} | {mean_bg_std[M]:.1f} |

![LBGI and HBGI per-record boxplots](figures/risk_indices.png)

## 8. AZT1D insulin / carb behaviour panel

![AZT1D vs Sim insulin / carb panel](figures/azt1d_event_panel.png)

| Quantity | AZT1D | T1DMSIM | Δ (Sim − AZT1D) |
|---|---:|---:|---:|
| Mean basal rate (U/hr)            | {az_pool['basal_pooled_mean']:.2f} | {sim_pool['basal_pooled_mean']:.2f} | {sim_pool['basal_pooled_mean'] - az_pool['basal_pooled_mean']:+.2f} |
| Median basal rate (U/hr)          | {az_pool['basal_pooled_median']:.2f} | {sim_pool['basal_pooled_median']:.2f} | {sim_pool['basal_pooled_median'] - az_pool['basal_pooled_median']:+.2f} |
| Basal P10–P90 spread (U/hr)       | {az_pool['basal_pooled_p10']:.2f} – {az_pool['basal_pooled_p90']:.2f} | {sim_pool['basal_pooled_p10']:.2f} – {sim_pool['basal_pooled_p90']:.2f} | — |
| Carbs / day (g, per-subject mean) | {az_pool['carbs_per_day_mean']:.1f} | {sim_pool['carbs_per_day_mean']:.1f} | {sim_pool['carbs_per_day_mean'] - az_pool['carbs_per_day_mean']:+.1f} |
| Total insulin / day (U)           | {az_pool['total_insulin_per_day_mean']:.1f} | {sim_pool['total_insulin_per_day_mean']:.1f} | {sim_pool['total_insulin_per_day_mean'] - az_pool['total_insulin_per_day_mean']:+.1f} |

| Quantity | AZT1D (per-subject mean) |
|---|---:|
| User-initiated boluses / day        | {az_pool['boluses_per_day_mean']:.2f} |
| Meal boluses / day                  | {az_pool['meal_boluses_per_day_mean']:.2f} |
| Correction-only boluses / day       | {az_pool['correction_boluses_per_day_mean']:.2f} |
| Mean carbs / meal (g)               | {az_pool['mean_carb_per_meal_mean']:.1f} |
| Correction-unit share of total bolus | {az_pool['correction_unit_share_pct']:.1f}% |

| Bolus type (whole AZT1D pool, incl. AID-driven) | Count |
|---|---:|
{chr(10).join(f"| {bt} | {cnt:,} |" for bt, cnt in sorted(az_pool['bolus_type_counts'].items(), key=lambda kv: -kv[1])[:10])}

| Device-mode time share | AZT1D % |
|---|---:|
{chr(10).join(f"| {k} | {v:.1f} |" for k, v in sorted(az_pool.get('device_mode_pct', {}).items(), key=lambda kv: -kv[1]))}

## 9. Side-by-side summary

| Quantity | T1DMSIM | OhioT1DM | ShanghaiT1DM | AZT1D | Sim − Ohio | Sim − Shang | Sim − AZT1D |
|---|---:|---:|---:|---:|---:|---:|---:|
{syn_row("Pooled mean BG (mg/dL)",       pm[M]['mean'],     pm[O]['mean'],     pm[S]['mean'],     pm[A]['mean'])}
{syn_row("Pooled median BG (mg/dL)",     pm[M]['median'],   pm[O]['median'],   pm[S]['median'],   pm[A]['median'])}
{syn_row("Pooled std (mg/dL)",           pm[M]['std'],      pm[O]['std'],      pm[S]['std'],      pm[A]['std'])}
{syn_row("Pooled CV (%)",                pm[M]['cv_pct'],   pm[O]['cv_pct'],   pm[S]['cv_pct'],   pm[A]['cv_pct'])}
{syn_row("Pooled skewness",              pm[M]['skew'],     pm[O]['skew'],     pm[S]['skew'],     pm[A]['skew'], fmt=".2f")}
{syn_row("Pooled excess kurtosis",       pm[M]['excess_kurt'], pm[O]['excess_kurt'], pm[S]['excess_kurt'], pm[A]['excess_kurt'], fmt=".2f")}
{syn_row("Pooled p99 (mg/dL)",           pp[M]['p99'],      pp[O]['p99'],      pp[S]['p99'],      pp[A]['p99'])}
{syn_row("GMI (per-record mean)",        sm[M]['GMI']['mean'],     sm[O]['GMI']['mean'],     sm[S]['GMI']['mean'],     sm[A]['GMI']['mean'], fmt=".2f")}
{syn_row("LBGI (per-record mean)",       sm[M]['LBGI']['mean'],    sm[O]['LBGI']['mean'],    sm[S]['LBGI']['mean'],    sm[A]['LBGI']['mean'], fmt=".2f")}
{syn_row("HBGI (per-record mean)",       sm[M]['HBGI']['mean'],    sm[O]['HBGI']['mean'],    sm[S]['HBGI']['mean'],    sm[A]['HBGI']['mean'], fmt=".2f")}
{syn_row("TIR % (per-record mean)",      sm[M]['TIR_pct']['mean'], sm[O]['TIR_pct']['mean'], sm[S]['TIR_pct']['mean'], sm[A]['TIR_pct']['mean'])}
{syn_row("TBR1 % (per-record mean)",     sm[M]['TBR1_pct']['mean'], sm[O]['TBR1_pct']['mean'], sm[S]['TBR1_pct']['mean'], sm[A]['TBR1_pct']['mean'], fmt=".2f")}
{syn_row("TBR2 % (per-record mean)",     sm[M]['TBR2_pct']['mean'], sm[O]['TBR2_pct']['mean'], sm[S]['TBR2_pct']['mean'], sm[A]['TBR2_pct']['mean'], fmt=".2f")}
{syn_row("TAR1 % (per-record mean)",     sm[M]['TAR1_pct']['mean'], sm[O]['TAR1_pct']['mean'], sm[S]['TAR1_pct']['mean'], sm[A]['TAR1_pct']['mean'])}
{syn_row("TAR2 % (per-record mean)",     sm[M]['TAR2_pct']['mean'], sm[O]['TAR2_pct']['mean'], sm[S]['TAR2_pct']['mean'], sm[A]['TAR2_pct']['mean'])}
{syn_row("MAGE (mg/dL)",                 sm[M]['mage']['mean'],    sm[O]['mage']['mean'],    sm[S]['mage']['mean'],    sm[A]['mage']['mean'])}
{syn_row("CONGA-1h (mg/dL)",             sm[M]['conga_1h']['mean'],sm[O]['conga_1h']['mean'],sm[S]['conga_1h']['mean'],sm[A]['conga_1h']['mean'])}
{syn_row("CONGA-4h (mg/dL)",             sm[M]['conga_4h']['mean'],sm[O]['conga_4h']['mean'],sm[S]['conga_4h']['mean'],sm[A]['conga_4h']['mean'])}
{syn_row("MODD (mg/dL)",                 sm[M]['modd']['mean'],    sm[O]['modd']['mean'],    sm[S]['modd']['mean'],    sm[A]['modd']['mean'])}
{syn_row("Hypo episodes / day",          sm[M]['hypo_count_per_day']['mean'], sm[O]['hypo_count_per_day']['mean'], sm[S]['hypo_count_per_day']['mean'], sm[A]['hypo_count_per_day']['mean'], fmt=".2f")}
{syn_row("Severe-hypo eps / day",        sm[M]['severe_hypo_count_per_day']['mean'], sm[O]['severe_hypo_count_per_day']['mean'], sm[S]['severe_hypo_count_per_day']['mean'], sm[A]['severe_hypo_count_per_day']['mean'], fmt=".2f")}
{syn_row("Hyper episodes / day",         sm[M]['hyper_count_per_day']['mean'], sm[O]['hyper_count_per_day']['mean'], sm[S]['hyper_count_per_day']['mean'], sm[A]['hyper_count_per_day']['mean'], fmt=".2f")}
{syn_row("Severe-hyper eps / day",       sm[M]['severe_hyper_count_per_day']['mean'], sm[O]['severe_hyper_count_per_day']['mean'], sm[S]['severe_hyper_count_per_day']['mean'], sm[A]['severe_hyper_count_per_day']['mean'], fmt=".2f")}
{syn_row("Hypo p90 duration (min)",      sm[M]['hypo_p90_min']['mean'],   sm[O]['hypo_p90_min']['mean'],   sm[S]['hypo_p90_min']['mean'],   sm[A]['hypo_p90_min']['mean'])}
{syn_row("Hyper p90 duration (min)",     sm[M]['hyper_p90_min']['mean'],  sm[O]['hyper_p90_min']['mean'],  sm[S]['hyper_p90_min']['mean'],  sm[A]['hyper_p90_min']['mean'])}
{syn_row("Hypo recovery median (min)",   rec[M]['median'],    rec[O]['median'],    rec[S]['median'],    rec[A]['median'])}

| Pooled distance | Sim vs Ohio | Sim vs Shanghai | Sim vs AZT1D | Ohio–Shang | Ohio–AZT1D | Shang–AZT1D |
|---|---:|---:|---:|---:|---:|---:|
| Wasserstein-1 (mg/dL) | {distances['Sim_vs_Ohio']['wasserstein']:.1f} | {distances['Sim_vs_Shanghai']['wasserstein']:.1f} | {distances['Sim_vs_AZT1D']['wasserstein']:.1f} | {distances['Ohio_vs_Shanghai']['wasserstein']:.1f} | {distances['Ohio_vs_AZT1D']['wasserstein']:.1f} | {distances['Shanghai_vs_AZT1D']['wasserstein']:.1f} |
| KS statistic | {distances['Sim_vs_Ohio']['ks_stat']:.3f} | {distances['Sim_vs_Shanghai']['ks_stat']:.3f} | {distances['Sim_vs_AZT1D']['ks_stat']:.3f} | {distances['Ohio_vs_Shanghai']['ks_stat']:.3f} | {distances['Ohio_vs_AZT1D']['ks_stat']:.3f} | {distances['Shanghai_vs_AZT1D']['ks_stat']:.3f} |

{ext_section_md}
"""
    # Column-alignment padding inside f-string bold markers (e.g. `**{x:>5.2f}**`)
    # leaks spaces between the asterisks and the value, which GitHub's renderer
    # then refuses to bold. Trim whitespace flush against the markers — but only
    # *inside* a bold pair, so adjacent `**bold** *italic*` constructs aren't
    # smashed together into `**bold***italic*`.
    md = re.sub(r"\*\*[ \t]*([^*\n]+?)[ \t]*\*\*", r"**\1**", md)
    with open(path, "w") as f:
        f.write(md)


def fig_gap_score(ext, path):
    """Diverging bar of the standardised sim-vs-real gap z per metric."""
    gaps = ext["gap_score"]
    items = [(k, v) for k, v in gaps.items() if np.isfinite(v.get("z", np.nan))]
    items.sort(key=lambda kv: kv[1]["z"])
    labels = [k for k, _ in items]
    zs = [v["z"] for _, v in items]

    def _c(z):
        a = abs(z)
        return "#2ca02c" if a < 1 else ("#ff7f0e" if a < 2 else "#d62728")
    colors = [_c(z) for z in zs]
    fig, ax = plt.subplots(figsize=(9, max(4, 0.42 * len(labels))))
    y = np.arange(len(labels))
    ax.barh(y, zs, color=colors)
    ax.axvspan(-1, 1, color="#2ca02c", alpha=0.08)
    for x in (-2, -1, 1, 2):
        ax.axvline(x, color="grey", lw=0.6, ls=":")
    ax.axvline(0, color="black", lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("standardised gap  z = (sim − mean_real) / sd_between_real")
    ax.set_title("Where the simulator sits inside vs outside the real cohorts' own spread\n"
                 "|z|<1 within (green) · 1–2 edge (amber) · >2 outside (red)")
    ax.grid(alpha=0.25, axis="x")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_band_transitions(ext, path):
    """5×5 glycemic-band transition-probability heatmaps, one per cohort."""
    labels = es.BAND_LABELS
    fig, axes = plt.subplots(1, len(ORDER), figsize=(4.3 * len(ORDER), 4.2))
    im = None
    for ax, n in zip(axes, ORDER):
        m = np.asarray(ext["transitions"][n]["matrix"], dtype=float)
        im = ax.imshow(m, vmin=0, vmax=1, cmap="magma")
        ax.set_xticks(range(5)); ax.set_yticks(range(5))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_title(n)
        ax.set_xlabel("to"); ax.set_ylabel("from")
        for i in range(5):
            for j in range(5):
                if m[i, j] >= 0.005:
                    ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center",
                            color="white" if m[i, j] < 0.6 else "black", fontsize=6)
    fig.suptitle("Glycemic-band transition probabilities (15-min grid)", fontweight="bold")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label="P(to | from)")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Extended statistics (Section 10) — see diff/extended_stats.py
# ============================================================================
# Curated metrics for the standardised strength/weakness gap score. Each entry
# extracts a single scalar per cohort. "hib" = higher-is-*neither*; the score is
# direction-agnostic (|z| measures distance into/out of the real spread).
def _pooled_tir(cohorts, n, key):
    d = cohorts[n]["pooled_bg"]
    return time_in_ranges(d).get(key, float("nan"))


def compute_extended(cohorts, pooled_moments, pooled_risk):
    """All Section-12 extended statistics.

    Reuses the audited core primitives (mage/conga/autocorr_lags/
    episode_durations) through extended_stats.cadence_fair, so the cadence-fair
    block introduces no new metric math — only a common-grid decimation.
    """
    ext = {}

    # --- 12.1 Cadence-fair variability (common 15-min grid) ---
    cf = {}
    cf_keys = ["mage", "delta_std", "conga_1h", "acf_30m", "acf_60m",
               "acf_120m", "hypo_per_day", "hyper_per_day"]
    for n in ORDER:
        vals = [es.cadence_fair(bg, cohorts[n]["series_step"], mage, conga,
                                autocorr_lags, episode_durations)
                for bg in cohorts[n]["series"]]
        cf[n] = {k: float(np.nanmean([v[k] for v in vals])) if vals
                 else float("nan") for k in cf_keys}
    ext["cadence_fair"] = cf

    # --- 12.2 Extra two-sample distances ---
    # CvM/AD statistics scale with n, so every pair is evaluated at one common
    # arm size (the smallest pooled cohort, capped) to be cross-comparable.
    pairs = [("Sim", "Ohio"), ("Sim", "Shanghai"), ("Sim", "AZT1D"),
             ("Ohio", "Shanghai"), ("Ohio", "AZT1D"), ("Shanghai", "AZT1D")]
    common_n = int(min(min(int(np.sum(~np.isnan(cohorts[n]["pooled_bg"])))
                           for n in ORDER), 60000))
    ext["extra_distances"] = {
        f"{a}_vs_{b}": es.extra_distances(cohorts[a]["pooled_bg"],
                                          cohorts[b]["pooled_bg"],
                                          common_n=common_n)
        for a, b in pairs}

    # --- 12.3 Temporal structure + band transitions ---
    temporal = {}
    transitions = {}
    for n in ORDER:
        step = cohorts[n]["series_step"]
        sd1s, sd2s, ratios, sents, cents, dfas, tcounts = [], [], [], [], [], [], []
        for bg in cohorts[n]["series"]:
            grid, eff = es.to_common_grid(bg, step, 15)
            p = es.poincare_sd(grid, eff)
            sd1s.append(p["sd1"]); sd2s.append(p["sd2"]); ratios.append(p["sd_ratio"])
            se = es.spectral_entropy(grid, eff)
            sents.append(se["spectral_entropy"]); cents.append(se["spectral_centroid_cph"])
            dfas.append(es.dfa_alpha(grid, eff))
            tcounts.append(es.band_transition(bg, step, 15)["counts"])
        temporal[n] = {
            "sd1": float(np.nanmean(sd1s)) if sd1s else float("nan"),
            "sd2": float(np.nanmean(sd2s)) if sd2s else float("nan"),
            "sd_ratio": float(np.nanmean(ratios)) if ratios else float("nan"),
            "spectral_entropy": float(np.nanmean(sents)) if sents else float("nan"),
            "spectral_centroid_cph": float(np.nanmean(cents)) if cents else float("nan"),
            "dfa_alpha": float(np.nanmean(dfas)) if dfas else float("nan"),
            "acf_efold_min": es.acf_efold_min(cohorts[n]["pooled_acf"]),
        }
        transitions[n] = es.pool_transition_counts(tcounts, 15)
    ext["temporal"] = temporal
    ext["transitions"] = transitions
    ext["transition_distance"] = {
        n: es.transition_matrix_distance(transitions["Sim"]["matrix"],
                                         transitions[n]["matrix"])
        for n in REAL_ORDER}

    # --- 12.4 Cross-seed bootstrap CIs on headline pooled scalars ---
    boot_stats = {
        "mean": np.mean,
        "std": np.std,
        "TIR_pct": lambda x: 100.0 * np.mean((x >= 70) & (x <= 180)),
        "LBGI": lambda x: kovatchev_risk(x)[0],
    }
    boot = {}
    for n in ORDER:
        recs = [r[~np.isnan(r)] for r in cohorts[n]["series"]]
        boot[n] = {name: es.bootstrap_pooled(recs, fn, n_boot=400, rng_seed=0)
                   for name, fn in boot_stats.items()}
    ext["bootstrap"] = boot

    # --- 12.5 Standardised strength/weakness gap score ---
    # metric label -> {cohort: scalar}
    metric_vals = {}
    for mk in ("mean", "std", "cv_pct", "skew", "excess_kurt"):
        metric_vals[mk] = {n: pooled_moments[n].get(mk, float("nan")) for n in ORDER}
    for band, lab in (("TIR_pct", "TIR%"), ("TBR1_pct", "TBR1%"),
                      ("TBR2_pct", "TBR2%"), ("TAR2_pct", "TAR2%")):
        metric_vals[lab] = {n: _pooled_tir(cohorts, n, band) for n in ORDER}
    metric_vals["LBGI"] = {n: pooled_risk[n]["LBGI_pooled"] for n in ORDER}
    metric_vals["HBGI"] = {n: pooled_risk[n]["HBGI_pooled"] for n in ORDER}
    for mk in ("mage", "delta_std", "conga_1h", "hypo_per_day", "hyper_per_day"):
        metric_vals[f"cf_{mk}"] = {n: cf[n][mk] for n in ORDER}
    for mk in ("sd_ratio", "spectral_entropy", "dfa_alpha", "acf_efold_min"):
        metric_vals[mk] = {n: temporal[n][mk] for n in ORDER}

    gaps = {}
    for label, vals in metric_vals.items():
        gs = es.gap_score(vals["Sim"], [vals["Ohio"], vals["Shanghai"], vals["AZT1D"]])
        gs["sim"] = vals["Sim"]
        gs["ohio"] = vals["Ohio"]; gs["shanghai"] = vals["Shanghai"]; gs["azt1d"] = vals["AZT1D"]
        gaps[label] = gs
    ext["gap_score"] = gaps
    return ext


def _build_extended_section(ext, cohorts):
    """Section 12 markdown from the extended-stats dict. Observational only."""
    O, S, A, M = "Ohio", "Shanghai", "AZT1D", "Sim"
    cf = ext["cadence_fair"]
    tp = ext["temporal"]
    ed = ext["extra_distances"]
    bt = ext["transitions"]
    boot = ext["bootstrap"]
    gaps = ext["gap_score"]

    def r(v, f=".2f"):
        return "n/a" if v is None or not np.isfinite(v) else f"{v:{f}}"

    def cf_row(label, key, f=".2f"):
        return (f"| {label} | {r(cf[O][key], f)} | {r(cf[S][key], f)} | "
                f"{r(cf[A][key], f)} | **{r(cf[M][key], f)}** |")

    def ed_row(label, key, f=".3f"):
        return "| " + label + " | " + " | ".join(
            r(ed[p][key], f) for p in ("Sim_vs_Ohio", "Sim_vs_Shanghai",
                                       "Sim_vs_AZT1D", "Ohio_vs_Shanghai",
                                       "Ohio_vs_AZT1D", "Shanghai_vs_AZT1D")) + " |"

    def tp_row(label, key, f=".3f"):
        return (f"| {label} | {r(tp[O][key], f)} | {r(tp[S][key], f)} | "
                f"{r(tp[A][key], f)} | **{r(tp[M][key], f)}** |")

    def efold(n):
        v = tp[n]["acf_efold_min"]
        if not np.isfinite(v):
            return ">24 h"
        return f"{v/60:.1f} h" if v >= 60 else f"{v:.0f} min"

    def dwell_row(band):
        return (f"| {band} | " + " | ".join(
            r(bt[n]["dwell_min"].get(band, float('nan')), ".0f") for n in ORDER) + " |")

    def boot_row(label, key, f=".1f"):
        def cell(n):
            b = boot[n][key]
            return f"{r(b['point'], f)} [{r(b['lo'], f)}, {r(b['hi'], f)}]"
        return f"| {label} | {cell(O)} | {cell(S)} | {cell(A)} | **{cell(M)}** |"

    # gap-score table — ALL metrics (never silently drop a non-finite z, which
    # would hide a degenerate-SD or never-e-folding outside-envelope weakness).
    def _znum(v):
        z = v.get("z")
        return z if (z is not None and np.isfinite(z)) else None

    def zfmt(v):
        z = v.get("z")
        if z is None or (isinstance(z, float) and np.isnan(z)):
            return "n/a"
        if not np.isfinite(z):
            return "+∞" if z > 0 else "−∞"
        return f"{z:+.2f}"

    gitems = sorted(gaps.items(),
                    key=lambda kv: -(abs(_znum(kv[1])) if _znum(kv[1]) is not None
                                     else (1e18 if not kv[1]["within_envelope"] else -1.0)))

    def g_row(label, v):
        env = "yes" if v["within_envelope"] else "**no**"
        return (f"| {label} | {r(v['sim'])} | {r(v['ohio'])} | {r(v['shanghai'])} | "
                f"{r(v['azt1d'])} | {zfmt(v)} | {env} |")
    gap_rows = "\n".join(g_row(k, v) for k, v in gitems)
    td = ext["transition_distance"]

    return f"""## 10. Extended statistics

### 10.1 Cadence-fair variability (common 15-min grid)

| Metric (15-min grid) | OhioT1DM | ShanghaiT1DM | AZT1D | T1DMSIM |
|---|---:|---:|---:|---:|
{cf_row("MAGE (mg/dL)", "mage", ".1f")}
{cf_row("Δ-BG SD, 15-min (mg/dL)", "delta_std", ".2f")}
{cf_row("CONGA-1h (mg/dL)", "conga_1h", ".2f")}
{cf_row("ACF @ 30 min", "acf_30m", ".3f")}
{cf_row("ACF @ 60 min", "acf_60m", ".3f")}
{cf_row("ACF @ 120 min", "acf_120m", ".3f")}
{cf_row("Hypo eps / day", "hypo_per_day", ".2f")}
{cf_row("Hyper eps / day", "hyper_per_day", ".2f")}

### 10.2 Additional two-sample distances (pooled BG)

| Distance | Sim vs Ohio | Sim vs Shang | Sim vs AZT1D | Ohio–Shang | Ohio–AZT1D | Shang–AZT1D |
|---|---:|---:|---:|---:|---:|---:|
{ed_row("Energy distance", "energy", ".3f")}
{ed_row("Cramér–von Mises", "cramer_von_mises", ".1f")}
{ed_row("Anderson–Darling", "anderson_darling", ".1f")}
{ed_row("Total variation", "total_variation", ".3f")}
{ed_row("Hellinger", "hellinger", ".3f")}
{ed_row("Histogram overlap", "overlap", ".3f")}

### 10.3 Temporal structure (common 15-min grid)

| Metric | OhioT1DM | ShanghaiT1DM | AZT1D | T1DMSIM |
|---|---:|---:|---:|---:|
{tp_row("Poincaré SD1 (mg/dL)", "sd1", ".2f")}
{tp_row("Poincaré SD2 (mg/dL)", "sd2", ".1f")}
{tp_row("Poincaré SD1/SD2", "sd_ratio", ".3f")}
{tp_row("Spectral entropy (0–1)", "spectral_entropy", ".3f")}
{tp_row("Spectral centroid (cyc/h)", "spectral_centroid_cph", ".3f")}
{tp_row("DFA α (Hurst)", "dfa_alpha", ".3f")}
| ACF e-folding (1/e) | {efold(O)} | {efold(S)} | {efold(A)} | **{efold(M)}** |

![Band transition heatmaps](figures/band_transitions.png)

| Band | OhioT1DM | ShanghaiT1DM | AZT1D | T1DMSIM |
|---|---:|---:|---:|---:|
{dwell_row("TBR2")}
{dwell_row("TBR1")}
{dwell_row("TIR")}
{dwell_row("TAR1")}
{dwell_row("TAR2")}

| Transition-matrix distance from Sim (Frobenius) | OhioT1DM | ShanghaiT1DM | AZT1D |
|---|---:|---:|---:|
| Distance | {r(td[O], ".3f")} | {r(td[S], ".3f")} | {r(td[A], ".3f")} |

### 10.4 Cross-seed bootstrap 95% CIs

| Statistic | OhioT1DM | ShanghaiT1DM | AZT1D | T1DMSIM |
|---|---|---|---|---|
{boot_row("Pooled mean (mg/dL)", "mean", ".1f")}
{boot_row("Pooled std (mg/dL)", "std", ".1f")}
{boot_row("TIR % (70–180)", "TIR_pct", ".1f")}
{boot_row("LBGI", "LBGI", ".2f")}

### 10.5 Standardised strength / weakness gap score

![Standardised gap score](figures/gap_score.png)

| Metric | Sim | Ohio | Shanghai | AZT1D | z | within envelope (±{es.ENVELOPE_REL_TOL:.0%} of span) |
|---|---:|---:|---:|---:|---:|:--:|
{gap_rows}
"""


# ============================================================================
# Main
# ============================================================================
def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-seeds", type=int, default=100,
                    help="number of simulator seeds to pool (default 100)")
    ap.add_argument("--days", type=int, default=70,
                    help="simulated days per seed (default 70)")
    ap.add_argument("--warmup-h", type=int, default=24,
                    help="warm-up hours discarded per seed (default 24)")
    cfg = ap.parse_args()

    print("Loading OhioT1DM…")
    ohio_p = load_ohio_patients()
    ohio = assemble_cohort("Ohio", sorted(ohio_p.items()), regularize_bg, step_min=5)
    print(f"  {len(ohio['per'])} patients")

    print("Loading ShanghaiT1DM…")
    shang_p = load_shanghai_patients()
    shang = assemble_cohort("Shanghai", sorted(shang_p.items()),
                            regularize_bg_15min, step_min=15)
    print(f"  {len(shang['per'])} records")

    print("Loading AZT1D…")
    az_p = load_azt1d_patients()
    azt1d = assemble_cohort("AZT1D", sorted(az_p.items()),
                            regularize_bg, step_min=5)
    print(f"  {len(azt1d['per'])} subjects")

    sim_items, sim_raw = assemble_sim(n_seeds=cfg.n_seeds, days=cfg.days,
                                      warmup_h=cfg.warmup_h)
    sim = assemble_cohort("Sim", sim_items, trivial_regularize_5min, step_min=5)
    print(f"  {len(sim['per'])} simulator runs")

    cohorts = {"Ohio": ohio, "Shanghai": shang, "AZT1D": azt1d, "Sim": sim}

    # AZT1D event log: insulin / carb / device-mode columns that the other two
    # real corpora do not expose at this granularity.
    print("Loading AZT1D event log…")
    az_events = load_azt1d_events()
    az_event_stats = azt1d_event_summary(az_events)
    sim_event_stats = sim_event_summary(sim_raw)

    # Unexplained-excursion analysis (§6.3) — reuse the already-loaded CGM and
    # event logs plus the sim raw runs (the simulator is not exercised again).
    print("Unexplained-excursion analysis…")
    ohio_ev = load_ohio_events()
    az_lists = {k: azt1d_event_lists(v) for k, v in az_events.items()}
    u_ohio = unexplained_cohort_real(
        "OhioT1DM", sorted(ohio_p.items()),
        lambda pid: ohio_ev.get(pid, ([], [], [])))
    u_az = unexplained_cohort_real(
        "AZT1D", sorted(az_p.items()),
        lambda sid: az_lists.get(sid, ([], [], [])))
    u_sim = unexplained_cohort_sim(sim_raw)
    unexpl_stats = {
        "Ohio": unexplained_summarize(u_ohio),
        "AZT1D": unexplained_summarize(u_az),
        "Sim": unexplained_summarize(u_sim),
        "params": {"amp_mg_dl": EXC_AMP, "reversal_mg_dl": EXC_REVERSAL,
                   "carb_min_g": EXC_CARB_MIN, "ins_min_U": EXC_INS_MIN},
    }
    real_excs = u_ohio["excs"] + u_az["excs"]
    real_days = (sum(r[6] for r in u_ohio["records"])
                 + sum(r[6] for r in u_az["records"]))
    sim_days = sum(r[6] for r in u_sim["records"])

    # Distribution distances — all sim/real pairs + every real/real baseline.
    distances = {}
    bins = np.arange(0, 601, 5)  # full support so Shanghai's tail is not dropped
    pair_list = [("Sim", "Ohio"), ("Sim", "Shanghai"), ("Sim", "AZT1D"),
                 ("Ohio", "Shanghai"), ("Ohio", "AZT1D"),
                 ("Shanghai", "AZT1D")]
    for a, b in pair_list:
        distances[f"{a}_vs_{b}"] = distribution_distances(
            cohorts[a]["pooled_bg"], cohorts[b]["pooled_bg"], bins=bins)

    # Pooled distribution moments + percentiles
    pooled_moments = {}
    pooled_percentiles = {}
    pooled_risk = {}
    for n in ORDER:
        d = cohorts[n]["pooled_bg"]
        pooled_moments[n] = central_moments(d)
        pooled_percentiles[n] = percentile_row(d)
        lbgi, hbgi = kovatchev_risk(d)
        pooled_risk[n] = {"LBGI_pooled": lbgi, "HBGI_pooled": hbgi,
                          "J_index_pooled": j_index(d),
                          "M_value_pooled": m_value(d)}

    # Per-record cohort summaries
    cohort_summaries = {n: cohort_summary(cohorts[n]["per"]) for n in ORDER}

    # Extended statistics (§10): cadence-fair, extra distances, temporal
    # structure, bootstrap CIs, gap score.
    print("Computing extended statistics…")
    ext = compute_extended(cohorts, pooled_moments, pooled_risk)

    # Figures
    print("Generating figures…")
    fig_pdf_pooled(cohorts, os.path.join(FIGS, "pdf_pooled.png"))
    fig_cdf_pooled(cohorts, os.path.join(FIGS, "cdf_pooled.png"))
    fig_diurnal_envelope(cohorts, os.path.join(FIGS, "diurnal_envelope.png"))
    fig_diurnal_envelope_median(cohorts, os.path.join(FIGS, "diurnal_envelope_median.png"))
    fig_weekday_heatmaps(cohorts, os.path.join(FIGS, "weekday_heatmap.png"))
    fig_weekday_heatmaps_median(cohorts, os.path.join(FIGS, "weekday_heatmap_median.png"))
    fig_acf(cohorts, os.path.join(FIGS, "acf.png"))
    fig_delta_distribution(cohorts, os.path.join(FIGS, "delta_distribution.png"))
    fig_risk_indices(cohorts, os.path.join(FIGS, "risk_indices.png"))
    fig_per_patient_scatter(cohorts, os.path.join(FIGS, "per_patient_scatter.png"))
    fig_episode_durations(cohorts, os.path.join(FIGS, "episode_durations.png"))
    fig_variability_metrics(cohorts, os.path.join(FIGS, "variability_metrics.png"))
    fig_pct_table_figure(cohorts, os.path.join(FIGS, "percentile_curves.png"))
    fig_qq(cohorts, os.path.join(FIGS, "qq.png"))
    fig_recovery(cohorts, os.path.join(FIGS, "recovery_time.png"))
    fig_diurnal_lines(cohorts, os.path.join(FIGS, "diurnal_lines.png"))
    fig_class_balance(cohorts, os.path.join(FIGS, "class_balance.png"))
    fig_azt1d_events(az_event_stats, sim_event_stats,
                     os.path.join(FIGS, "azt1d_event_panel.png"))
    fig_unexplained_summary(
        [unexpl_stats["Ohio"], unexpl_stats["AZT1D"], unexpl_stats["Sim"]],
        [("Ohio+AZT1D (real)", real_excs, real_days, "#1f77b4"),
         ("T1DMSIM", u_sim["excs"], sim_days, COL["Sim"])],
        os.path.join(FIGS, "unexplained_summary.png"))
    fig_unexplained_gallery([u_ohio, u_az],
                            os.path.join(FIGS, "unexplained_gallery.png"))
    fig_gap_score(ext, os.path.join(FIGS, "gap_score.png"))
    fig_band_transitions(ext, os.path.join(FIGS, "band_transitions.png"))

    # Recovery-time summary per cohort
    recov_summaries = {n: recovery_summary(cohorts[n]["recov_times"]) for n in ORDER}

    # Persist computed stats
    payload = {
        "datasets": {n: {
            "n_records": len(cohorts[n]["per"]),
            "step_min": cohorts[n]["step_min"],
            "total_days": sum(p["days"] for p in cohorts[n]["per"]),
            "pooled_moments": pooled_moments[n],
            "pooled_percentiles": pooled_percentiles[n],
            "pooled_risk": pooled_risk[n],
            "pooled_acf": cohorts[n]["pooled_acf"],
            "diurnal_mean": cohorts[n]["diurnal_mean"].tolist(),
            "diurnal_std": cohorts[n]["diurnal_std"].tolist(),
            "diurnal_median": cohorts[n]["diurnal_median"].tolist(),
            "diurnal_p25": cohorts[n]["diurnal_p25"].tolist(),
            "diurnal_p75": cohorts[n]["diurnal_p75"].tolist(),
            "summary": cohort_summaries[n],
            "n_hypo": len(cohorts[n]["hypo_durs"]),
            "n_hyper": len(cohorts[n]["hyper_durs"]),
            "n_severe_hypo": len(cohorts[n]["severe_hypo_durs"]),
            "n_severe_hyper": len(cohorts[n]["severe_hyper_durs"]),
            "recovery": recov_summaries[n],
        } for n in ORDER},
        "distances": distances,
        # Strip the raw per-sample value arrays (used only for the histogram
        # figures, already rendered above) before serialization — they are
        # ~0.6M floats/seed and bloated stats.json to 64 MB, scaling linearly
        # with the seed count.
        "azt1d_events": {k: v for k, v in az_event_stats.items()
                         if k not in ("basal_values", "carb_values")},
        "sim_events": {k: v for k, v in sim_event_stats.items()
                       if k not in ("basal_values", "carb_values")},
        "unexplained_excursions": unexpl_stats,
        "extended": ext,
    }
    out = os.path.join(DIFF, "stats.json")
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, default=float)
    print(f"Wrote {out}")

    # Templated markdown report
    report_path = os.path.join(DIFF, "README.md")
    write_report_md(cohorts, distances, pooled_moments, pooled_percentiles,
                    pooled_risk, cohort_summaries, recov_summaries,
                    az_event_stats, sim_event_stats, unexpl_stats, ext,
                    report_path)
    print(f"Wrote {report_path}")
    print(f"Figures in {FIGS}")


if __name__ == "__main__":
    main()
