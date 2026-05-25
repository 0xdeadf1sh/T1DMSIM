"""Compare simulator output against the OhioT1DM real-world CGM dataset.

Prints a pooled per-metric comparison plus a top-gaps ranking. Useful for
checking how close the synthetic data is to a real T1D cohort after a
tuning change.

Run from the repo root with the project venv:

    venv/bin/python scripts/compare_to_ohio.py

Reads OhioT1DM data from ./ohiot1dm/*.xml (gitignored, non-redistributable).
"""
import os
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO_ROOT)
import simulator as sim  # noqa: E402

OHIO_DIR = os.path.join(REPO_ROOT, 'ohiot1dm')
STEP_MIN = 5
N_SEEDS = 40
DAYS = 70
WARMUP_HOURS = 24  # sim starts at initial_bg=140 and needs ~24h to reach steady state


def load_ohio() -> dict:
    """Parse all Ohio XML files, return {patient_id: [(ts, bg), ...]}."""
    pts: dict = defaultdict(list)
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
    return pts


def regularize(rows, step_min: int = STEP_MIN):
    """Resample irregularly-sampled CGM to a fixed 5-min grid; gaps >30 min → NaN."""
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
            if dt1 > 30 * 60:
                v = float('nan')
            elif dt1 == 0:
                v = vals[i]
            else:
                w = dt0 / dt1
                v = vals[i] + w * (vals[i + 1] - vals[i])
        else:
            v = vals[i]
        out_t.append(cur)
        out_v.append(v)
        cur = cur + timedelta(minutes=step_min)
    return np.array(out_t), np.array(out_v)


# ---------- Stat blocks ----------
def basic_stats(bg):
    bg = bg[~np.isnan(bg)]
    if len(bg) == 0:
        return {}
    return {
        'mean': float(np.mean(bg)),
        'median': float(np.median(bg)),
        'std': float(np.std(bg)),
        'cv_%': float(100 * np.std(bg) / np.mean(bg)),
        'gmi': float(3.31 + 0.02392 * np.mean(bg)),
        'p10': float(np.percentile(bg, 10)),
        'p25': float(np.percentile(bg, 25)),
        'p75': float(np.percentile(bg, 75)),
        'p90': float(np.percentile(bg, 90)),
        'min': float(np.min(bg)),
        'max': float(np.max(bg)),
    }


def clinical_ranges(bg):
    bg = bg[~np.isnan(bg)]
    n = len(bg)
    if n == 0:
        return {}
    return {
        'TBR2_<54': float(100 * np.sum(bg < 54) / n),
        'TBR1_54-70': float(100 * np.sum((bg >= 54) & (bg < 70)) / n),
        'TIR_70-180': float(100 * np.sum((bg >= 70) & (bg <= 180)) / n),
        'TAR1_180-250': float(100 * np.sum((bg > 180) & (bg <= 250)) / n),
        'TAR2_>250': float(100 * np.sum(bg > 250) / n),
    }


def episodes(bg, lo: float, hi: float, step_min: int = STEP_MIN, min_minutes: int = 15):
    """Return list of (start_idx, duration_min) where bg in [lo, hi)."""
    bg = bg.copy()
    bg[np.isnan(bg)] = (lo + hi) / 2  # bridge NaN gaps so they don't split episodes
    inrange = (bg >= lo) & (bg < hi)
    eps = []
    i = 0
    n = len(inrange)
    while i < n:
        if inrange[i]:
            j = i
            while j < n and inrange[j]:
                j += 1
            dur = (j - i) * step_min
            if dur >= min_minutes:
                eps.append((i, dur))
            i = j
        else:
            i += 1
    return eps


def episode_stats(eps):
    if not eps:
        return {'n': 0, 'median': 0.0, 'p90': 0.0, 'max': 0.0, 'mean': 0.0}
    durs = np.array([d for _, d in eps])
    return {
        'n': int(len(eps)),
        'median': float(np.median(durs)),
        'p90': float(np.percentile(durs, 90)),
        'max': float(np.max(durs)),
        'mean': float(np.mean(durs)),
    }


def diurnal(bg, times):
    by_hour = defaultdict(list)
    for t, v in zip(times, bg):
        if not np.isnan(v):
            by_hour[t.hour].append(v)
    return np.array([float(np.mean(by_hour[h])) if by_hour[h] else float('nan')
                     for h in range(24)])


def acf(bg, lag: int):
    bg = bg[~np.isnan(bg)]
    if len(bg) < lag + 2:
        return float('nan')
    x = bg[:-lag]
    y = bg[lag:]
    mx, my = float(np.mean(x)), float(np.mean(y))
    num = float(np.sum((x - mx) * (y - my)))
    den = float(np.sqrt(np.sum((x - mx) ** 2) * np.sum((y - my) ** 2)))
    return num / den if den > 0 else float('nan')


def delta_stats(bg):
    bg = bg[~np.isnan(bg)]
    if len(bg) < 2:
        return {}
    d = np.diff(bg)
    return {
        'd_std': float(np.std(d)),
        'd_p90_abs': float(np.percentile(np.abs(d), 90)),
        'd_max_abs': float(np.max(np.abs(d))),
        'd_acf1': acf(d, lag=1),
    }


def stat_block(times, bg) -> dict:
    out: dict = {**basic_stats(bg), **clinical_ranges(bg)}
    hypo = episode_stats(episodes(bg, 0, 70))
    hyper = episode_stats(episodes(bg, 180, 1000))
    severe = episode_stats(episodes(bg, 0, 54))
    days = (len(bg) * STEP_MIN) / (60 * 24)
    out['hypo_n_per_day'] = hypo['n'] / days if days else 0.0
    out['hyper_n_per_day'] = hyper['n'] / days if days else 0.0
    out['hypo_median_min'] = hypo['median']
    out['hypo_p90_min'] = hypo['p90']
    out['hypo_max_min'] = hypo['max']
    out['hyper_median_min'] = hyper['median']
    out['hyper_p90_min'] = hyper['p90']
    out['hyper_max_min'] = hyper['max']
    out['severe_max_min'] = severe['max']
    out['severe_mean_min'] = severe['mean']
    out['severe_n_per_day'] = severe['n'] / days if days else 0.0
    out['diurnal'] = diurnal(bg, times)
    out.update(delta_stats(bg))
    return out


# ---------- Pipelines ----------
def run_ohio() -> dict:
    pts = load_ohio()
    per = {}
    for pid, rows in pts.items():
        t, bg = regularize(rows)
        if len(bg) < 200:
            continue
        per[pid] = stat_block(t, bg)
    return per


def run_sim(n_seeds: int = N_SEEDS, days: int = DAYS) -> dict:
    per = {}
    steps_per_hour = 60 // STEP_MIN
    warmup_steps = WARMUP_HOURS * steps_per_hour
    steps = days * 24 * steps_per_hour
    for seed in range(n_seeds):
        s = sim.T1DMSimulator(seed=seed, initial_bg=140.0)
        for _ in range(warmup_steps):
            s.generate()
        bg = np.zeros(steps)
        for i in range(steps):
            bg[i] = float(s.generate()['bg_observed'])  # pyright: ignore[reportArgumentType]
        start = datetime(2025, 1, 6, 0, 0)
        times = np.array([start + timedelta(minutes=STEP_MIN * i) for i in range(steps)])
        per[f'seed{seed}'] = stat_block(times, bg)
    return per


def pool(per: dict) -> dict:
    keys = [k for k in next(iter(per.values())).keys() if k != 'diurnal']
    out = {k: float(np.mean([p[k] for p in per.values() if not np.isnan(p[k])]))
           for k in keys}
    out['diurnal'] = np.nanmean(np.stack([p['diurnal'] for p in per.values()]), axis=0)
    return out


def fmt_compare(ohio: dict, simp: dict):
    print(f'\n{"Metric":<25} {"Ohio":>10} {"Sim":>10} {"Δ":>10}  {"Δ%":>8}')
    print('-' * 70)
    keys = [
        'mean', 'median', 'std', 'cv_%', 'gmi',
        'p10', 'p25', 'p75', 'p90', 'min', 'max',
        'TBR2_<54', 'TBR1_54-70', 'TIR_70-180', 'TAR1_180-250', 'TAR2_>250',
        'hypo_n_per_day', 'hypo_median_min', 'hypo_p90_min', 'hypo_max_min',
        'severe_n_per_day', 'severe_max_min', 'severe_mean_min',
        'hyper_n_per_day', 'hyper_median_min', 'hyper_p90_min', 'hyper_max_min',
        'd_std', 'd_p90_abs', 'd_max_abs', 'd_acf1',
    ]
    deltas = []
    for k in keys:
        o = ohio.get(k, float('nan'))
        s = simp.get(k, float('nan'))
        dpct = float('nan') if (o == 0 or np.isnan(o)) else 100 * (s - o) / abs(o)
        print(f'{k:<25} {o:>10.2f} {s:>10.2f} {s-o:>+10.2f}  {dpct:>+7.1f}%')
        deltas.append((k, o, s, dpct))
    print('-' * 70)

    print('\nDiurnal (mean BG by hour-of-day):')
    diff = ohio['diurnal'] - simp['diurnal']
    for h in range(24):
        bar = '#' * int(abs(diff[h]) / 2)
        sign = '+' if diff[h] >= 0 else '-'
        print(f'  {h:02d}h  Ohio={ohio["diurnal"][h]:6.1f}  Sim={simp["diurnal"][h]:6.1f}  Δ={sign}{bar}')

    print('\nTop gaps by |Δ%|:')
    ranked = sorted([(k, o, s, d) for k, o, s, d in deltas if not np.isnan(d)],
                    key=lambda x: -abs(x[3]))
    for k, o, s, d in ranked[:12]:
        print(f'  {k:<25} Ohio={o:8.2f}  Sim={s:8.2f}  Δ%={d:+7.1f}')


def main():
    print(f'Loading OhioT1DM from {OHIO_DIR}...')
    ohio = run_ohio()
    print(f'  {len(ohio)} patients regularised')

    print(f'Running simulator: {N_SEEDS} seeds × {DAYS}d...')
    simp = run_sim()
    print(f'  {len(simp)} seeds')

    fmt_compare(pool(ohio), pool(simp))


if __name__ == '__main__':
    main()
