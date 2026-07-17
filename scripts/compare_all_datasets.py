"""Three-way comparison pipeline: OhioT1DM, ShanghaiT1DM, T1DMSIM.

Exports loaders, regularisers, per-patient stat blocks, and the run_pipeline
aggregator consumed by scripts/generate_comparison_figures.py. Also runnable
directly to print pooled aggregate stats.

Reads from:
    ./datasets/ohiot1dm/*.xml                      (5-min Dexcom CGM, US adults)
    ./datasets/ShanghaiT1DM/Shanghai_T1DM/*.xls(x) (15-min, CN adults)
    ./datasets/AZT1D/CGM Records/Subject N/*.csv   (5-min Dexcom + AID pump,
                                                    Mayo Clinic Arizona)

Datasets live under datasets/ and are gitignored (non-redistributable).
"""
from __future__ import annotations

import os
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO_ROOT)
from simulator import T1DMSimulator  # noqa: E402

DATASETS_DIR = os.path.join(REPO_ROOT, 'datasets')
OHIO_DIR = os.path.join(DATASETS_DIR, 'ohiot1dm')
SHANG_DIR = os.path.join(DATASETS_DIR, 'ShanghaiT1DM', 'Shanghai_T1DM')
AZT1D_DIR = os.path.join(DATASETS_DIR, 'AZT1D', 'CGM Records')


# ---------- Loaders ----------
def load_ohio_patients() -> dict:
    """Return {patient_id: [(ts, bg_mg_dl), ...]} from the Ohio XML corpus."""
    pts: dict = defaultdict(list)
    if not os.path.isdir(OHIO_DIR):
        return {}
    for fn in os.listdir(OHIO_DIR):
        if not fn.endswith('.xml'):
            continue
        pid = fn.split('-')[0]
        try:
            tree = ET.parse(os.path.join(OHIO_DIR, fn))
        except ET.ParseError:
            continue
        for glu in tree.getroot().iter('glucose_level'):
            for ev in glu.iter('event'):
                ts = ev.get('ts')
                val = ev.get('value')
                if not (ts and val):
                    continue
                try:
                    t = datetime.strptime(ts, '%d-%m-%Y %H:%M:%S')
                except ValueError:
                    continue
                pts[pid].append((t, float(val)))
    for pid in pts:
        pts[pid].sort()
    return dict(pts)


def load_shanghai_patients() -> dict:
    """Return {record_id: [(ts, bg_mg_dl), ...]} from the Shanghai xls(x) corpus."""
    pts: dict = {}
    if not os.path.isdir(SHANG_DIR):
        return {}
    for fn in sorted(os.listdir(SHANG_DIR)):
        if fn.startswith('~') or not (fn.endswith('.xlsx') or fn.endswith('.xls')):
            continue
        rid = fn.rsplit('.', 1)[0]
        try:
            df = pd.read_excel(os.path.join(SHANG_DIR, fn))
        except Exception:
            continue
        if 'CGM (mg / dl)' not in df.columns or 'Date' not in df.columns:
            continue
        rows = []
        ts_col = pd.to_datetime(df['Date'], errors='coerce')
        bg_col = pd.to_numeric(df['CGM (mg / dl)'], errors='coerce')
        for ts, val in zip(ts_col, bg_col):
            if pd.isna(ts) or pd.isna(val):
                continue
            rows.append((ts.to_pydatetime(), float(val)))
        if rows:
            rows.sort()
            pts[rid] = rows
    return pts


def load_azt1d_patients() -> dict:
    """Return {subject_id: [(ts, bg_mg_dl), ...]} from the AZT1D CSV corpus.

    Only CGM samples are loaded here; insulin / carb / device-mode columns are
    loaded separately by `load_azt1d_events`.
    """
    pts: dict = {}
    if not os.path.isdir(AZT1D_DIR):
        return {}
    for sub in sorted(os.listdir(AZT1D_DIR)):
        sub_dir = os.path.join(AZT1D_DIR, sub)
        if not os.path.isdir(sub_dir):
            continue
        csvs = [f for f in os.listdir(sub_dir) if f.endswith('.csv')]
        if not csvs:
            continue
        path = os.path.join(sub_dir, csvs[0])
        try:
            header = pd.read_csv(path, nrows=0).columns
        except Exception:
            continue
        # Most subjects expose a bare 'CGM' column; Subject 14 alone names it
        # 'Readings (CGM / BGM)'. Match any header containing 'CGM' so a renamed
        # column cannot silently drop a subject (was excluding Subject 14, a
        # low-mean outlier, biasing the AZT1D reference up ~1.7 mg/dL).
        cgm_col = 'CGM' if 'CGM' in header else next(
            (c for c in header if 'CGM' in c), None)
        if cgm_col is None or 'EventDateTime' not in header:
            continue
        try:
            df = pd.read_csv(path, usecols=['EventDateTime', cgm_col])
        except Exception:
            continue
        ts_col = pd.to_datetime(df['EventDateTime'], errors='coerce')
        bg_col = pd.to_numeric(df[cgm_col], errors='coerce')
        rows = []
        for ts, val in zip(ts_col, bg_col):
            if pd.isna(ts) or pd.isna(val):
                continue
            rows.append((ts.to_pydatetime(), float(val)))
        if rows:
            rows.sort()
            pts[sub] = rows
    return pts


def load_azt1d_events() -> dict:
    """Return {subject_id: DataFrame} with the full AZT1D event log.

    Columns kept: EventDateTime, DeviceMode, BolusType, Basal,
    CorrectionDelivered, TotalBolusInsulinDelivered, FoodDelivered, CarbSize.
    Numeric columns are coerced; missing event fields stay NaN.
    """
    out: dict = {}
    if not os.path.isdir(AZT1D_DIR):
        return {}
    cols = ['EventDateTime', 'DeviceMode', 'BolusType', 'Basal',
            'CorrectionDelivered', 'TotalBolusInsulinDelivered',
            'FoodDelivered', 'CarbSize']
    for sub in sorted(os.listdir(AZT1D_DIR)):
        sub_dir = os.path.join(AZT1D_DIR, sub)
        if not os.path.isdir(sub_dir):
            continue
        csvs = [f for f in os.listdir(sub_dir) if f.endswith('.csv')]
        if not csvs:
            continue
        try:
            df = pd.read_csv(os.path.join(sub_dir, csvs[0]), usecols=cols)
        except Exception:
            continue
        df['EventDateTime'] = pd.to_datetime(df['EventDateTime'],
                                              errors='coerce')
        for c in ('Basal', 'CorrectionDelivered',
                  'TotalBolusInsulinDelivered', 'FoodDelivered', 'CarbSize'):
            df[c] = pd.to_numeric(df[c], errors='coerce')
        out[sub] = df
    return out


# ---------- Regularisation ----------
def _regularize_rows(rows, step_min: int, gap_min: int):
    """Resample irregular CGM to a fixed grid, NaN-bridging gaps > gap_min."""
    if not rows:
        return np.array([]), np.array([])
    times, vals = zip(*rows)
    times = list(times)
    vals = list(vals)
    out_t: list = []
    out_v: list = []
    cur = times[0]
    i = 0
    while cur <= times[-1]:
        while i + 1 < len(times) and times[i + 1] <= cur:
            i += 1
        if i + 1 < len(times):
            dt0 = (cur - times[i]).total_seconds()
            dt1 = (times[i + 1] - times[i]).total_seconds()
            if dt1 > gap_min * 60:
                v = float('nan')
            elif dt1 == 0:
                v = vals[i]
            else:
                # Snap to the nearer real sample rather than linearly
                # interpolating. Linear interpolation is a low-pass that shaves
                # ~5-7% off the real cohorts' short-scale variability (their
                # Dexcom timestamps drift off the grid), while the on-grid
                # simulator is passed through untouched — biasing every
                # high-frequency comparison (Δ-BG SD, short-lag ACF). Nearest
                # keeps both arms on the same footing.
                v = vals[i] if 2 * dt0 <= dt1 else vals[i + 1]
        else:
            v = vals[i]
        out_t.append(cur)
        out_v.append(v)
        cur = cur + timedelta(minutes=step_min)
    return np.array(out_t), np.array(out_v)


def regularize_bg(rows):
    """5-min grid; gaps over 30 min become NaN. Used for OhioT1DM."""
    return _regularize_rows(rows, step_min=5, gap_min=30)


def regularize_bg_15min(rows):
    """15-min grid; gaps over 60 min become NaN. Used for ShanghaiT1DM."""
    return _regularize_rows(rows, step_min=15, gap_min=60)


# ---------- Event rates (stubs — figure script does not display these but
# run_pipeline accepts a callback so we keep the signature). ----------
def ohio_event_rates(_pid: str) -> dict:
    return {}


def shanghai_event_rates(_rid: str) -> dict:
    return {}


# ---------- Stat blocks ----------
def basic_stats(bg) -> dict:
    bg = bg[~np.isnan(bg)]
    if len(bg) == 0:
        return {}
    return {
        'mean_BG': float(np.mean(bg)),
        'median_BG': float(np.median(bg)),
        'std_BG': float(np.std(bg)),
        'CV_pct': float(100 * np.std(bg) / np.mean(bg)),
        'GMI': float(3.31 + 0.02392 * np.mean(bg)),
    }


def clinical_ranges(bg) -> dict:
    bg = bg[~np.isnan(bg)]
    n = len(bg)
    if n == 0:
        return {}
    return {
        'TBR2_pct': float(100 * np.sum(bg < 54) / n),
        'TBR1_pct': float(100 * np.sum((bg >= 54) & (bg < 70)) / n),
        'TIR_pct': float(100 * np.sum((bg >= 70) & (bg <= 180)) / n),
        'TAR1_pct': float(100 * np.sum((bg > 180) & (bg <= 250)) / n),
        'TAR2_pct': float(100 * np.sum(bg > 250) / n),
    }


def variability_metrics(bg, step_min: int = 5) -> dict:
    # Diff first, then drop NaN-touching pairs. Filtering NaN first would
    # create artificial jumps across former gaps and inflate the std.
    if len(bg) < 2:
        return {f'd{step_min}min_std': 0.0, f'd{step_min}min_p90_abs': 0.0}
    d = np.diff(bg)
    d = d[~np.isnan(d)]
    if len(d) == 0:
        return {f'd{step_min}min_std': 0.0, f'd{step_min}min_p90_abs': 0.0}
    return {
        f'd{step_min}min_std': float(np.std(d)),
        f'd{step_min}min_p90_abs': float(np.percentile(np.abs(d), 90)),
    }


def episodes(bg, threshold: float, below: bool = True,
             step_min: int = 5, min_minutes: int = 15) -> list:
    """Return a flat list of episode durations (minutes) where bg crosses
    `threshold` (below=True for hypo, False for hyper). Bridges NaN gaps."""
    bg = bg.copy()
    if below:
        bg[np.isnan(bg)] = threshold + 1.0  # bridge NaN as in-range
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


def diurnal_means(times, bg) -> np.ndarray:
    """Mean BG by hour-of-day. Returns an array of 24."""
    by_hour: dict = defaultdict(list)
    for t, v in zip(times, bg):
        if not np.isnan(v):
            by_hour[t.hour].append(v)
    return np.array([float(np.mean(by_hour[h])) if by_hour[h] else float('nan')
                     for h in range(24)])


def aggregate(per: list) -> dict:
    """Population mean/std across per-record stat dicts."""
    if not per:
        return {}
    keys = set().union(*(p.keys() for p in per)) - {'rid'}
    out = {}
    for k in keys:
        vals = [p[k] for p in per if k in p and not np.isnan(p[k])]
        if vals:
            out[k] = (float(np.mean(vals)), float(np.std(vals)))
    return out


# ---------- Pipelines ----------
def run_pipeline(name: str, items, regularize_fn, event_rates_fn,
                 step_min: int = 5) -> dict:
    """Per-record stats + pooled aggregates for a real-cohort dataset."""
    per = []
    pooled = []
    diurnals = []
    hypo_durs: list = []
    hyper_durs: list = []
    for rid, rows in items:
        t, bg = regularize_fn(rows)
        if len(bg) < 200:
            continue
        rec = {
            'rid': str(rid),
            **basic_stats(bg),
            **clinical_ranges(bg),
            **variability_metrics(bg, step_min=step_min),
            **event_rates_fn(rid),
        }
        # Observed (non-NaN) coverage, NOT calendar span: NaN-bridged gap cells
        # carry no episodes, so counting them as exposure deflates per-day rates
        # for gappy real cohorts (~8% on Ohio) while the gapless sim is exact.
        days = (int(np.sum(~np.isnan(bg))) * step_min) / (60 * 24)
        h_d = episodes(bg, 70, below=True, step_min=step_min)
        H_d = episodes(bg, 180, below=False, step_min=step_min)
        rec['hypo_count_per_day'] = len(h_d) / days if days else 0.0
        rec['hyper_count_per_day'] = len(H_d) / days if days else 0.0
        per.append(rec)
        pooled.append(bg[~np.isnan(bg)])
        diurnals.append(diurnal_means(t, bg))
        hypo_durs.extend(h_d)
        hyper_durs.extend(H_d)
    return {
        'name': name,
        'per': per,
        'agg': aggregate(per),
        'pooled_bg': np.concatenate(pooled) if pooled else np.array([]),
        'diurnal': np.nanmean(np.stack(diurnals), axis=0)
                   if diurnals else np.full(24, np.nan),
        'hypo_durs': hypo_durs,
        'hyper_durs': hyper_durs,
    }


class TappedSimulator(T1DMSimulator):
    """T1DMSimulator alias — kept as a thin subclass so callers that want to
    instrument step-level outputs have a single hook point. Currently no
    additional behavior."""
    pass


def print_summary(name: str, d: dict):
    a = d['agg']
    g = lambda k: a[k][0] if k in a else float('nan')  # noqa: E731
    print(f'\n--- {name} (n={len(d["per"])}) ---')
    print(f'  mean BG   {g("mean_BG"):6.1f}')
    print(f'  GMI       {g("GMI"):6.2f}')
    print(f'  CV %      {g("CV_pct"):6.1f}')
    print(f'  TBR2      {g("TBR2_pct"):6.2f}')
    print(f'  TBR1      {g("TBR1_pct"):6.2f}')
    print(f'  TIR       {g("TIR_pct"):6.2f}')
    print(f'  TAR1      {g("TAR1_pct"):6.2f}')
    print(f'  TAR2      {g("TAR2_pct"):6.2f}')


def main(n_seeds: int = 30, days: int = 70):
    print('Loading OhioT1DM...')
    ohio_p = load_ohio_patients()
    ohio = run_pipeline('Ohio', sorted(ohio_p.items()),
                        regularize_bg, ohio_event_rates, step_min=5)
    print(f'  {len(ohio["per"])} patients')

    print('Loading ShanghaiT1DM...')
    shang_p = load_shanghai_patients()
    shang = run_pipeline('Shang', sorted(shang_p.items()),
                         regularize_bg_15min, shanghai_event_rates, step_min=15)
    print(f'  {len(shang["per"])} records')

    print(f'Running T1DMSIM: {n_seeds} seeds × {days}d (24h warmup discarded)...')
    sim_per = []
    sim_pooled = []
    sim_diurnals = []
    sim_hypo_durs: list = []
    sim_hyper_durs: list = []
    warmup_hours = 24
    for seed in range(n_seeds):
        s = TappedSimulator(seed=seed, initial_bg=120.0)
        s.generate_hours(warmup_hours)  # discard transient before steady state
        data = s.generate_hours(days * 24)
        bg = np.asarray(data['bg_observed'])
        t0 = datetime(2024, 1, 1)
        times = np.array([t0 + timedelta(minutes=5 * i) for i in range(len(bg))])
        rec = {
            'rid': str(seed),
            **basic_stats(bg),
            **clinical_ranges(bg),
            **variability_metrics(bg, step_min=5),
        }
        h_d = episodes(bg, 70, below=True, step_min=5)
        H_d = episodes(bg, 180, below=False, step_min=5)
        rec['hypo_count_per_day'] = len(h_d) / days
        rec['hyper_count_per_day'] = len(H_d) / days
        sim_per.append(rec)
        sim_pooled.append(bg[~np.isnan(bg)])
        sim_diurnals.append(diurnal_means(times, bg))
        sim_hypo_durs.extend(h_d)
        sim_hyper_durs.extend(H_d)
    sim = {
        'name': 'Sim',
        'per': sim_per,
        'agg': aggregate(sim_per),
        'pooled_bg': np.concatenate(sim_pooled),
        'diurnal': np.nanmean(np.stack(sim_diurnals), axis=0),
        'hypo_durs': sim_hypo_durs,
        'hyper_durs': sim_hyper_durs,
    }

    print_summary('OhioT1DM', ohio)
    print_summary('ShanghaiT1DM', shang)
    print_summary('T1DMSIM', sim)
    return ohio, shang, sim


if __name__ == '__main__':
    main()
