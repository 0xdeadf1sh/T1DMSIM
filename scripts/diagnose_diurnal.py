"""
Phase-1 diagnostic for the dinner-peak / nocturnal-low / late-dawn / hypo-count
chain. Decomposes 30 seeds × 70 days into per-hour means of BG, insulin, carbs,
HGO, and glucose_in/out; histograms hypo-episode start times; and partitions
the dawn curve on overnight BG so we can tell whether the late dawn peak is a
bounce-from-low artefact or an independent timing problem.

Usage: python scripts/diagnose_diurnal.py [--seeds N] [--days D]
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
from simulator import T1DMSimulator, DT_MINUTES


STEPS_PER_HOUR = 60 // DT_MINUTES        # 12
STEPS_PER_DAY = 24 * STEPS_PER_HOUR      # 288


def episode_starts(bg, threshold=70, below=True, min_minutes=15):
    """Return step indices where qualifying episodes begin."""
    mask = bg < threshold if below else bg > threshold
    out = []
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if (j - i) * DT_MINUTES >= min_minutes:
                out.append(i)
            i = j
        else:
            i += 1
    return out


def run_seed(seed, days, warmup_h):
    sim = T1DMSimulator(seed=seed, initial_bg=120.0)
    sim.generate_hours(warmup_h)
    return sim.generate_hours(days * 24)


def row(label, values, scale=1.0, width=5, decimals=1):
    body = "  ".join(f"{v*scale:>{width}.{decimals}f}" for v in values)
    return f"{label:<28}{body}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=30)
    ap.add_argument('--days', type=int, default=70)
    ap.add_argument('--warmup-hours', type=float, default=24.0)
    args = ap.parse_args()

    bg_sums = np.zeros(24)
    insulin_sums = np.zeros(24)
    basal_sums = np.zeros(24)
    bolus_sums = np.zeros(24)
    carb_sums = np.zeros(24)
    hgo_sums = np.zeros(24)
    gin_sums = np.zeros(24)
    gout_sums = np.zeros(24)
    counts = np.zeros(24)

    hypo_start_hours = np.zeros(24, dtype=int)
    hyper_start_hours = np.zeros(24, dtype=int)

    # Conditional on BG at 03:00
    bg_lowstart_sums = np.zeros(24)
    bg_highstart_sums = np.zeros(24)
    n_lowstart = 0
    n_highstart = 0
    LOWSTART_THRESH = 130.0  # BG at 03:00 below this counts as a "nocturnal-low" day

    # Dinner-window energy budget
    dwin_in, dwin_out, dwin_drop = [], [], []

    print(f"Running {args.seeds} seeds × {args.days} days "
          f"({args.warmup_hours}h warmup)...", flush=True)

    for seed in range(args.seeds):
        d = run_seed(seed, args.days, args.warmup_hours)
        bg = d['bg']
        hod_int = d['hour_of_day'].astype(int)

        # Per-hour accumulation (vectorized via np.add.at)
        np.add.at(bg_sums,      hod_int, bg)
        np.add.at(insulin_sums, hod_int, d['total_insulin'])
        np.add.at(basal_sums,   hod_int, d['basal_insulin'])
        np.add.at(bolus_sums,   hod_int, d['bolus_insulin'])
        np.add.at(carb_sums,    hod_int, d['total_carb'])
        np.add.at(hgo_sums,     hod_int, d['hgo'])
        np.add.at(gin_sums,     hod_int, d['glucose_in'])
        np.add.at(gout_sums,    hod_int, d['glucose_out'])
        np.add.at(counts,       hod_int, 1)

        for i in episode_starts(bg, threshold=70, below=True, min_minutes=15):
            hypo_start_hours[hod_int[i]] += 1
        for i in episode_starts(bg, threshold=180, below=False, min_minutes=15):
            hyper_start_hours[hod_int[i]] += 1

        # Conditional dawn curve + dinner window
        n_days = len(bg) // STEPS_PER_DAY
        for di in range(n_days):
            s = di * STEPS_PER_DAY
            e = s + STEPS_PER_DAY
            day_bg = bg[s:e]
            day_hod = hod_int[s:e]

            # BG at 03:00 (mean of that hour). Skip days that don't contain it.
            h03 = (day_hod == 3)
            if not h03.any():
                continue
            bg_at_03 = float(day_bg[h03].mean())

            # Build the day's 24-hour profile
            day_profile = np.zeros(24)
            for h in range(24):
                m = (day_hod == h)
                if m.any():
                    day_profile[h] = float(day_bg[m].mean())

            if bg_at_03 < LOWSTART_THRESH:
                bg_lowstart_sums += day_profile
                n_lowstart += 1
            else:
                bg_highstart_sums += day_profile
                n_highstart += 1

            # Dinner window: 19:00 of day di → 03:00 of day di+1
            evening = (day_hod >= 19)
            wi = float(d['glucose_in'][s:e][evening].sum())
            wo = float(d['glucose_out'][s:e][evening].sum())
            bg_21 = day_bg[(day_hod == 21)].mean() if (day_hod == 21).any() else np.nan

            if di + 1 < n_days:
                ns = (di + 1) * STEPS_PER_DAY
                ne = ns + STEPS_PER_DAY
                nxt_hod = hod_int[ns:ne]
                early = (nxt_hod < 3)
                wi += float(d['glucose_in'][ns:ne][early].sum())
                wo += float(d['glucose_out'][ns:ne][early].sum())
                h03n = (nxt_hod == 3)
                bg_03n = float(bg[ns:ne][h03n].mean()) if h03n.any() else np.nan
                if not (np.isnan(bg_21) or np.isnan(bg_03n)):
                    dwin_drop.append(bg_21 - bg_03n)

            dwin_in.append(wi)
            dwin_out.append(wo)

    bg_mean = bg_sums / counts
    ins_mean = insulin_sums / counts
    bas_mean = basal_sums / counts
    bol_mean = bolus_sums / counts
    carb_mean = carb_sums / counts
    hgo_mean = hgo_sums / counts
    gin_mean = gin_sums / counts
    gout_mean = gout_sums / counts

    header_hours = "  ".join(f"{h:>5d}" for h in range(24))

    print()
    print("=" * 90)
    print("Per-hour means (averaged across all (seed, day) steps)")
    print("=" * 90)
    print(f"{'hour':<28}{header_hours}")
    print(row('BG (mg/dL)', bg_mean))
    print(row('total_insulin (U/hr)', ins_mean, scale=STEPS_PER_HOUR, decimals=3))
    print(row('  basal (U/hr)', bas_mean, scale=STEPS_PER_HOUR, decimals=3))
    print(row('  bolus (U/hr)', bol_mean, scale=STEPS_PER_HOUR, decimals=3))
    print(row('total_carb (g/hr)', carb_mean, scale=STEPS_PER_HOUR, decimals=2))
    print(row('HGO (g/hr)', hgo_mean, scale=STEPS_PER_HOUR, decimals=2))
    print(row('glucose_in  (g/hr)', gin_mean, scale=STEPS_PER_HOUR, decimals=2))
    print(row('glucose_out (g/hr)', gout_mean, scale=STEPS_PER_HOUR, decimals=2))
    print(row('net (in − out, g/hr)', (gin_mean - gout_mean),
              scale=STEPS_PER_HOUR, decimals=2))

    total_hypo = int(hypo_start_hours.sum())
    total_hyper = int(hyper_start_hours.sum())
    print()
    print("=" * 90)
    print(f"Hypo-episode starts (<70, ≥15 min): {total_hypo} events "
          f"= {total_hypo / (args.seeds * args.days):.2f}/day")
    print(f"Hyper-episode starts (>180, ≥15 min): {total_hyper} events "
          f"= {total_hyper / (args.seeds * args.days):.2f}/day")
    print("=" * 90)
    print(f"{'hour':<28}{header_hours}")
    print(f"{'hypo starts':<28}" + "  ".join(f"{v:>5d}" for v in hypo_start_hours))
    print(f"{'hypo % of all':<28}"
          + "  ".join(f"{(v / max(1, total_hypo)) * 100:>5.1f}" for v in hypo_start_hours))
    print(f"{'hyper starts':<28}" + "  ".join(f"{v:>5d}" for v in hyper_start_hours))
    print()
    bins = [(0, 6, 'nocturnal 00-06'), (6, 12, 'morning  06-12'),
            (12, 18, 'afternoon 12-18'), (18, 24, 'evening  18-24')]
    for lo, hi, label in bins:
        h_frac = hypo_start_hours[lo:hi].sum() / max(1, total_hypo)
        H_frac = hyper_start_hours[lo:hi].sum() / max(1, total_hyper)
        print(f"  {label}: hypo {h_frac:5.1%}    hyper {H_frac:5.1%}")

    print()
    print("=" * 90)
    print(f"Conditional dawn curve — partition by BG at 03:00")
    print(f"  Low-start days (BG@03:00 < {LOWSTART_THRESH:.0f}):  {n_lowstart}")
    print(f"  High-start days (BG@03:00 ≥ {LOWSTART_THRESH:.0f}): {n_highstart}")
    print("=" * 90)
    print(f"{'hour':<28}{header_hours}")
    if n_lowstart:
        print(row('low-start BG', bg_lowstart_sums / n_lowstart))
    if n_highstart:
        print(row('high-start BG', bg_highstart_sums / n_highstart))
    if n_lowstart and n_highstart:
        diff = bg_lowstart_sums / n_lowstart - bg_highstart_sums / n_highstart
        print(row('low − high', diff))

    print()
    print("=" * 90)
    print("Dinner-to-nocturnal window  (19:00 → 03:00 next day)")
    print("=" * 90)
    if dwin_in and dwin_out:
        wi = np.array(dwin_in)
        wo = np.array(dwin_out)
        print(f"  glucose_in  / window:  mean={wi.mean():6.1f} g   std={wi.std():5.1f}")
        print(f"  glucose_out / window:  mean={wo.mean():6.1f} g   std={wo.std():5.1f}")
        print(f"  net (in − out):        mean={(wi-wo).mean():+6.1f} g   std={(wi-wo).std():5.1f}")
    if dwin_drop:
        dd = np.array(dwin_drop)
        print(f"  BG(21:00) − BG(03:00): mean={dd.mean():+6.1f} mg/dL   "
              f"std={dd.std():5.1f}   p10={np.percentile(dd,10):+.0f}   "
              f"p90={np.percentile(dd,90):+.0f}")


if __name__ == '__main__':
    main()
