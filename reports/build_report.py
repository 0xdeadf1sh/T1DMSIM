"""Comprehensive statistical comparison of T1DMSIM vs OhioT1DM vs ShanghaiT1DM.

Produces:
  reports/stats.json        — all computed numbers (used by REPORT.md)
  reports/figures/*.png     — figure set referenced from REPORT.md

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
  - cohort summary tables in JSON keyed by metric

All three datasets are read with the loaders in scripts/compare_all_datasets.py;
the simulator is exercised with the same warm-up convention as that script.
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sps

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPORTS = os.path.join(REPO_ROOT, "reports")
FIGS = os.path.join(REPORTS, "figures")
os.makedirs(FIGS, exist_ok=True)

sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from compare_all_datasets import (  # noqa: E402
    load_ohio_patients, load_shanghai_patients,
    regularize_bg, regularize_bg_15min,
)
from simulator import T1DMSimulator  # noqa: E402


# ----- palette -----
COL = {"Ohio": "#1f77b4", "Shanghai": "#ff7f0e", "Sim": "#d62728"}
ORDER = ["Ohio", "Shanghai", "Sim"]


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


def sample_entropy(x, m=2, r_frac=0.2, subsample=2500):
    """Sample entropy SampEn(m, r). r = r_frac * std(x). To keep this cheap on
    long traces we randomly subsample down to `subsample` points."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) > subsample:
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(len(x), subsample, replace=False))
        x = x[idx]
    n = len(x)
    if n < m + 2:
        return float("nan")
    r = r_frac * np.std(x)
    if r == 0:
        return float("nan")

    def phi(mm):
        # max-norm template matching, exclude self-match
        templates = np.lib.stride_tricks.sliding_window_view(x, mm)
        ntemp = len(templates)
        cnt = 0
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


def episode_recovery_time(bg, low_thresh=70, normal_thresh=80, step_min=5):
    """For each hypo episode, time from first sample <70 to first subsequent
    sample ≥80. Bridges NaN gaps as in-range."""
    bg = bg.copy()
    bg[np.isnan(bg)] = 120.0
    out = []
    n = len(bg)
    i = 0
    while i < n:
        if bg[i] < low_thresh:
            j = i
            while j < n and bg[j] < normal_thresh:
                j += 1
            dur = (j - i) * step_min
            out.append(dur)
            i = j
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
    diurnals_mean = []
    diurnals_std = []
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
        days = (len(bg) * step_min) / (60 * 24)
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
        rec["sample_entropy"] = sample_entropy(bg)
        per.append(rec)
        pooled.append(bg_clean)
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
        diurnals_mean.append(m24)
        diurnals_std.append(s24)
        hypo_durs.extend(h)
        hyper_durs.extend(H)
        severe_hypo_durs.extend(sh)
        sev_hyper_durs.extend(SH)
        recov_times.extend(episode_recovery_time(bg, step_min=step_min))
    # Weekday × hour 7×24 matrix
    wd_grid = np.full((7, 24), np.nan)
    for (wd, h), arr in weekday_hour.items():
        wd_grid[wd, h] = float(np.mean(arr))
    cohort = {
        "name": name,
        "step_min": step_min,
        "per": per,
        "pooled_bg": np.concatenate(pooled) if pooled else np.array([]),
        "pooled_delta": np.concatenate(delta_pooled) if delta_pooled else np.array([]),
        "diurnal_mean": np.nanmean(np.stack(diurnals_mean), axis=0) if diurnals_mean else np.full(24, np.nan),
        "diurnal_std": np.nanmean(np.stack(diurnals_std), axis=0) if diurnals_std else np.full(24, np.nan),
        "wd_grid": wd_grid,
        "pooled_acf": {L: (float(np.mean(v)) if v else float("nan")) for L, v in pooled_acf.items()},
        "lags_min": lags_min,
        "hypo_durs": hypo_durs,
        "hyper_durs": hyper_durs,
        "severe_hypo_durs": severe_hypo_durs,
        "severe_hyper_durs": sev_hyper_durs,
        "recov_times": recov_times,
    }
    return cohort


def assemble_sim(n_seeds=30, days=70, warmup_h=24):
    print(f"Running T1DMSIM: {n_seeds} seeds × {days}d ({warmup_h}h warmup)…")
    items = []
    for seed in range(n_seeds):
        s = T1DMSimulator(seed=seed, initial_bg=120.0)
        s.generate_hours(warmup_h)
        d = s.generate_hours(days * 24)
        bg = np.asarray(d["bg_observed"], dtype=float)
        t0 = datetime(2024, 1, 1)
        rows = [(t0 + timedelta(minutes=5 * i), float(bg[i])) for i in range(len(bg))]
        items.append((str(seed), rows))
    return items


def trivial_regularize_5min(rows):
    """Sim rows are already on a 5-min grid; pass through."""
    times = np.array([r[0] for r in rows])
    vals = np.array([r[1] for r in rows], dtype=float)
    return times, vals


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
        bins = np.arange(40, 401, 5)
    ha, _ = np.histogram(a, bins=bins, density=True)
    hb, _ = np.histogram(b, bins=bins, density=True)
    js = jensen_shannon(ha, hb)
    return {"ks_stat": float(ks_stat), "ks_p": float(ks_p),
            "wasserstein": w, "js_div": js}


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
    for k in keys:
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


def fig_weekday_heatmaps(cohorts, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    vmin, vmax = 90, 220
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
    labels = {"Ohio": "Ohio (Δ5m)", "Shanghai": "Shanghai (Δ15m)", "Sim": "T1DMSIM (Δ5m)"}
    for n in ORDER:
        d = cohorts[n]["pooled_delta"]
        if len(d) == 0:
            continue
        ax.hist(d, bins=bins, density=True, histtype="step", lw=2.0,
                color=COL[n], label=labels[n])
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
    for ax, key, title in [(axes[0], "LBGI", "LBGI — low BG risk index"),
                           (axes[1], "HBGI", "HBGI — high BG risk index")]:
        data = []
        for n in ORDER:
            vals = [p[key] for p in cohorts[n]["per"]
                    if key in p and not np.isnan(p[key])]
            data.append(vals)
        bp = ax.boxplot(data, labels=ORDER, patch_artist=True, widths=0.55,
                        medianprops=dict(color="black", lw=2))
        for patch, n in zip(bp["boxes"], ORDER):
            patch.set_facecolor(COL[n])
            patch.set_alpha(0.6)
        for i, vals in enumerate(data, start=1):
            for v in vals:
                ax.plot(i + np.random.uniform(-0.07, 0.07), v, ".",
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
        bp = ax.boxplot(data, labels=ORDER, patch_artist=True, widths=0.55,
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
        bp = ax.boxplot(data, labels=ORDER, patch_artist=True, widths=0.55,
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
    """Quantile-Quantile plots Sim vs Ohio and Sim vs Shanghai."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    qs = np.linspace(0.01, 0.99, 200)
    sim_q = np.quantile(cohorts["Sim"]["pooled_bg"], qs)
    for ax, ref in zip(axes, ("Ohio", "Shanghai")):
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
    bp = ax.boxplot(data, labels=ORDER, patch_artist=True, widths=0.55,
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


# ============================================================================
# Main
# ============================================================================
def main():
    print("Loading OhioT1DM…")
    ohio_p = load_ohio_patients()
    ohio = assemble_cohort("Ohio", sorted(ohio_p.items()), regularize_bg, step_min=5)
    print(f"  {len(ohio['per'])} patients")

    print("Loading ShanghaiT1DM…")
    shang_p = load_shanghai_patients()
    shang = assemble_cohort("Shanghai", sorted(shang_p.items()),
                            regularize_bg_15min, step_min=15)
    print(f"  {len(shang['per'])} records")

    sim_items = assemble_sim(n_seeds=30, days=70, warmup_h=24)
    sim = assemble_cohort("Sim", sim_items, trivial_regularize_5min, step_min=5)
    print(f"  {len(sim['per'])} simulator runs")

    cohorts = {"Ohio": ohio, "Shanghai": shang, "Sim": sim}

    # Distribution distances
    distances = {}
    bins = np.arange(40, 401, 5)
    for pair in [("Sim", "Ohio"), ("Sim", "Shanghai"), ("Ohio", "Shanghai")]:
        a, b = pair
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

    # Figures
    print("Generating figures…")
    fig_pdf_pooled(cohorts, os.path.join(FIGS, "pdf_pooled.png"))
    fig_cdf_pooled(cohorts, os.path.join(FIGS, "cdf_pooled.png"))
    fig_diurnal_envelope(cohorts, os.path.join(FIGS, "diurnal_envelope.png"))
    fig_weekday_heatmaps(cohorts, os.path.join(FIGS, "weekday_heatmap.png"))
    fig_acf(cohorts, os.path.join(FIGS, "acf.png"))
    fig_delta_distribution(cohorts, os.path.join(FIGS, "delta_distribution.png"))
    fig_risk_indices(cohorts, os.path.join(FIGS, "risk_indices.png"))
    fig_per_patient_scatter(cohorts, os.path.join(FIGS, "per_patient_scatter.png"))
    fig_episode_durations(cohorts, os.path.join(FIGS, "episode_durations.png"))
    fig_variability_metrics(cohorts, os.path.join(FIGS, "variability_metrics.png"))
    fig_pct_table_figure(cohorts, os.path.join(FIGS, "percentile_curves.png"))
    fig_qq(cohorts, os.path.join(FIGS, "qq.png"))
    fig_recovery(cohorts, os.path.join(FIGS, "recovery_time.png"))

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
            "summary": cohort_summaries[n],
            "n_hypo": len(cohorts[n]["hypo_durs"]),
            "n_hyper": len(cohorts[n]["hyper_durs"]),
            "n_severe_hypo": len(cohorts[n]["severe_hypo_durs"]),
            "n_severe_hyper": len(cohorts[n]["severe_hyper_durs"]),
        } for n in ORDER},
        "distances": distances,
    }
    out = os.path.join(REPORTS, "stats.json")
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, default=float)
    print(f"Wrote {out}")
    print(f"Figures in {FIGS}")


if __name__ == "__main__":
    main()
