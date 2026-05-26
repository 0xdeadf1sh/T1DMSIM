"""
Phase-1.5 diagnostic: skill × IR decomposition of the diurnal BG profile.

Confirms (or refutes) the two-mode hypothesis from diagnose_diurnal.py:
  mode 1 — low-IR / well-bolused patients crash overnight from a dinner-bolus
           tail (the "low-start" 67% of day-records)
  mode 2 — high-IR / under-bolused patients stay stuck high overnight
           (the "high-start" 33%)

Splits the 30 seeds into a 2×2 by s3 (dosing competence) × IR factor, then
prints per-bucket hour-of-day BG means, hypo episodes/day, BG at 21:00 and
03:00, and a basal-titration ratio (basal_dose / ideal_basal). If mode 2
maps cleanly to one bucket (expected: high-IR + low-s3), under-bolus bias /
basal-titration are the right Phase-2 levers for that cohort; if it spans
buckets, mode 2 is a global problem and behavioral code shouldn't be touched.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
from simulator import (
    T1DMSimulator, DT_MINUTES,
    BODY_WEIGHT_MEAN_KG, HGO_BASE_GRAMS_PER_HOUR,
)


STEPS_PER_HOUR = 60 // DT_MINUTES
STEPS_PER_DAY = 24 * STEPS_PER_HOUR


def episode_starts(bg, threshold=70, below=True, min_minutes=15):
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


def hour_means(values, hod_int):
    out = np.zeros(24)
    for h in range(24):
        m = (hod_int == h)
        if m.any():
            out[h] = float(values[m].mean())
    return out


def run_seed(seed, days, warmup_h):
    sim = T1DMSimulator(seed=seed, initial_bg=120.0)
    sim.generate_hours(warmup_h)
    d = sim.generate_hours(days * 24)
    return sim.patient, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=30)
    ap.add_argument('--days', type=int, default=70)
    ap.add_argument('--warmup-hours', type=float, default=24.0)
    args = ap.parse_args()

    patients = []
    curves = []
    hypos_per_day = []
    bg21 = []
    bg03 = []
    bg_means = []

    print(f"Running {args.seeds} seeds × {args.days} days "
          f"({args.warmup_hours}h warmup)...", flush=True)

    for seed in range(args.seeds):
        p, d = run_seed(seed, args.days, args.warmup_hours)
        bg = d['bg']
        hod = d['hour_of_day'].astype(int)

        curve = hour_means(bg, hod)
        n_hypo = len(episode_starts(bg, threshold=70, below=True, min_minutes=15))

        patients.append(p)
        curves.append(curve)
        hypos_per_day.append(n_hypo / args.days)
        bg21.append(curve[21])
        bg03.append(curve[3])
        bg_means.append(float(bg.mean()))

    curves = np.array(curves)
    hypos_per_day = np.array(hypos_per_day)
    bg21 = np.array(bg21)
    bg03 = np.array(bg03)
    bg_means = np.array(bg_means)

    s1 = np.array([p.dietary_discipline for p in patients])
    s2 = np.array([p.attentiveness for p in patients])
    s3 = np.array([p.dosing_competence for p in patients])
    s4 = np.array([p.lifestyle_consistency for p in patients])
    ir = np.array([p.insulin_resistance_factor for p in patients])
    is_base = np.array([p.is_base for p in patients])
    icr = np.array([p.icr for p in patients])
    basal = np.array([p.basal_dose for p in patients])
    weight = np.array([p.body_weight_kg for p in patients])

    # Ideal basal from generate_patient() formula
    weight_factor = weight / BODY_WEIGHT_MEAN_KG
    ideal_basal = HGO_BASE_GRAMS_PER_HOUR * 24.0 * weight_factor * is_base / icr
    titration = basal / ideal_basal

    s3_med = float(np.median(s3))
    ir_med = float(np.median(ir))

    buckets = [
        ('low-s3 / low-IR  ',  (s3 < s3_med)  & (ir < ir_med)),
        ('low-s3 / high-IR ',  (s3 < s3_med)  & (ir >= ir_med)),
        ('high-s3 / low-IR ', (s3 >= s3_med) & (ir < ir_med)),
        ('high-s3 / high-IR', (s3 >= s3_med) & (ir >= ir_med)),
    ]

    header_hours = "  ".join(f"{h:>5d}" for h in range(24))

    print()
    print("=" * 100)
    print(f"Per-seed summary (s3 median = {s3_med:.3f}, IR median = {ir_med:.3f})")
    print("=" * 100)
    print(f"{'seed':>4} {'s1':>5} {'s3':>5} {'IR':>5} {'is_b':>5} {'ICR':>5} "
          f"{'basal':>6} {'ideal':>6} {'tit%':>5} "
          f"{'meanBG':>6} {'BG21':>5} {'BG03':>5} {'hypo/d':>6}")
    for i in range(args.seeds):
        print(f"{i:>4} {s1[i]:5.2f} {s3[i]:5.2f} {ir[i]:5.2f} "
              f"{is_base[i]:5.2f} {icr[i]:5.1f} "
              f"{basal[i]:6.1f} {ideal_basal[i]:6.1f} {titration[i]*100:5.0f} "
              f"{bg_means[i]:6.0f} {bg21[i]:5.0f} {bg03[i]:5.0f} "
              f"{hypos_per_day[i]:6.2f}")

    print()
    print("=" * 100)
    print("Per-bucket BG curves (hour-of-day mean across the seeds in the bucket)")
    print("=" * 100)
    print(f"{'bucket':<20}  n  " + header_hours)
    for label, mask in buckets:
        n = int(mask.sum())
        if n == 0:
            continue
        avg = curves[mask].mean(axis=0)
        body = "  ".join(f"{v:>5.0f}" for v in avg)
        print(f"{label:<20} {n:>2}  {body}")

    print()
    print("=" * 100)
    print("Per-bucket summary")
    print("=" * 100)
    print(f"{'bucket':<20} {'n':>3} {'mean BG':>8} {'BG21':>6} {'BG03':>6} "
          f"{'21−03':>6} {'hypo/d':>7} {'titration':>10} {'mean s3':>8} {'mean IR':>8}")
    for label, mask in buckets:
        n = int(mask.sum())
        if n == 0:
            print(f"{label:<20} {n:>3}  (empty)")
            continue
        print(f"{label:<20} {n:>3} "
              f"{bg_means[mask].mean():>8.1f} "
              f"{bg21[mask].mean():>6.0f} "
              f"{bg03[mask].mean():>6.0f} "
              f"{(bg21[mask] - bg03[mask]).mean():>+6.0f} "
              f"{hypos_per_day[mask].mean():>7.2f} "
              f"{titration[mask].mean()*100:>9.1f}% "
              f"{s3[mask].mean():>8.2f} "
              f"{ir[mask].mean():>8.2f}")

    # Correlation canaries
    print()
    print("=" * 100)
    print("Correlations across seeds (Pearson r)")
    print("=" * 100)
    pairs = [
        ('s1 (diet)         vs hypo/d', s1, hypos_per_day),
        ('s3 (dosing)       vs hypo/d', s3, hypos_per_day),
        ('IR                vs hypo/d', ir, hypos_per_day),
        ('titration         vs hypo/d', titration, hypos_per_day),
        ('s3 (dosing)       vs mean BG', s3, bg_means),
        ('IR                vs mean BG', ir, bg_means),
        ('titration         vs mean BG', titration, bg_means),
        ('IR                vs BG@21:00', ir, bg21),
        ('IR                vs BG@03:00', ir, bg03),
        ('titration         vs BG@03:00', titration, bg03),
    ]
    for label, x, y in pairs:
        if np.std(x) > 0 and np.std(y) > 0:
            r = float(np.corrcoef(x, y)[0, 1])
        else:
            r = float('nan')
        print(f"  {label}: r = {r:+.3f}")


if __name__ == '__main__':
    main()
