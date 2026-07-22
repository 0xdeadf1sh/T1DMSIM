"""Meal-excursion comparison: T1DMSIM vs. a UVA/Padova patient dosed for its
own physiology.

The companion `compare_uva_padova.py` replays T1DMSIM's insulin *verbatim* into
UVA/Padova; because T1DMSIM's boluses are sized by its own aggressive
insulin-to-carb ratio, that over-covers the UVA virtual patients' gentler carb
ratios and crushes them into sustained hypoglycaemia — a gross level mismatch
that swamps any finer comparison.

Here the inputs are recalibrated for UVA/Padova: the two engines share only the
**meal schedule** (time + grams that T1DMSIM generates per seed); each then
doses insulin for its own physiology — T1DMSIM as it normally would, and the
UVA/Padova patient on its steady-state basal plus a `grams / CR` meal bolus
(its own Quest-table carb ratio) and a bounded correction. With both engines in
a normal operating range, we isolate and compare the **post-meal excursions**
themselves: amplitude, time-to-peak, area, recovery, and — peak-normalised, so
amplitude no longer confounds it — excursion *shape*.

Writes `uva_padova/excursions.json`, excursion + gallery figures under
`uva_padova/figures/`, and a regenerated `uva_padova/EXCURSIONS.md`.

Run (reference engine installed without its RL extras):
    pip install --no-deps simglucose>=0.2.11
    venv/bin/python uva_padova/compare_excursions.py            # full run
    venv/bin/python uva_padova/compare_excursions.py --quick    # fast smoke run
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
FIGS = os.path.join(HERE, "figures")
sys.path.insert(0, REPO_ROOT)

import simulator as S  # noqa: E402
from padova_engine import PadovaPatient, patient_names  # noqa: E402

DT = S.DT_MINUTES
GROUP = "adult"
N_SEEDS = 40
DAYS = 14
WARMUP_HOURS = 48
MAX_WORKERS = max(1, (os.cpu_count() or 4) // 2)

# Excursion-extraction parameters.
WIN_MIN = 180                      # excursion window after the meal (3 h)
WIN = WIN_MIN // DT
MEAL_MIN_G = 25.0                  # only analyse meals at least this large
ISO_PRE_MIN = 120                  # require a quiet window before the meal
ISO_PRE = ISO_PRE_MIN // DT
ISO_POST_MIN = 120                 # ...and after it, so the rise+peak is clean
ISO_POST = ISO_POST_MIN // DT
ISO_OTHER_G = 20.0                 # ...with no other meal this big inside either
RET_THRESH = 15.0                  # "recovered" = within this of baseline
SIZE_BINS = [(25, 40), (40, 55), (55, 70), (70, 200)]

C_OURS = "#1f77b4"
C_UVA = "#d62728"
C_MEAL = "#2ca02c"


# ---------------------------------------------------------------------------
# One paired sample: our behavioural trace + UVA self-dosed on shared meals
# ---------------------------------------------------------------------------
def capture_ours(seed: int, days: int, initial_bg: float):
    """Run T1DMSIM; return (sim, true-BG per step, food-meal grams per step)."""
    sim = S.T1DMSimulator(seed=seed, initial_bg=initial_bg)
    meal_g: dict = {}
    original = sim.inject_curve

    def hook(values, start_idx, curve_type, label=""):
        if curve_type == "carb":   # food meals only — exclude rescue/correction carbs
            meal_g[int(start_idx)] = meal_g.get(int(start_idx), 0.0) + float(np.sum(values))
        return original(values, start_idx, curve_type, label)

    sim.inject_curve = hook  # type: ignore[method-assign]
    n = days * 24 * 60 // DT
    bg = np.empty(n)
    for i in range(n):
        bg[i] = float(sim.generate()["bg"])
    return sim, bg, meal_g


def run_pair(seed: int, days: int):
    name = patient_names(GROUP)[seed % len(patient_names(GROUP))]
    pad = PadovaPatient(name)
    _, bg_ours, meal_g = capture_ours(seed, days, pad.init_gsub)
    n = len(bg_ours)
    meals_min = [(step * DT, g) for step, g in meal_g.items()]
    gsub = pad.replay_self_dosed(meals_min, n * DT)
    bg_uva = gsub[::DT][:n]
    return {
        "seed": seed, "uva_name": name, "cr": pad.cr, "cf": pad.cf,
        "bg_ours": bg_ours, "bg_uva": bg_uva,
        "meal_steps": sorted(meal_g), "meal_g": [meal_g[s] for s in sorted(meal_g)],
    }


def _worker(args):
    return run_pair(*args)


# ---------------------------------------------------------------------------
# Excursion extraction
# ---------------------------------------------------------------------------
def detect_meals(meal_steps, meal_g_list, n_steps):
    """Isolated, large-enough meals with a full window ahead. Returns [(step, g)]."""
    g_by = dict(zip(meal_steps, meal_g_list))
    out = []
    for s in meal_steps:
        g = g_by[s]
        if g < MEAL_MIN_G or s < ISO_PRE or s + WIN >= n_steps:
            continue
        if any(k != s and g_by.get(k, 0.0) >= ISO_OTHER_G
               for k in range(s - ISO_PRE, s + ISO_POST + 1)):
            continue
        out.append((s, g))
    return out


def excursion(bg, s):
    """Features of the excursion that begins at step ``s`` (relative to baseline)."""
    base = float(bg[s])
    seg = bg[s:s + WIN + 1].astype(float) - base
    pk = int(np.argmax(seg))
    dbg = float(seg[pk])
    auc = float(np.sum(np.clip(seg, 0.0, None)) * DT)
    ret = WIN_MIN
    for k in range(pk, len(seg)):
        if seg[k] <= RET_THRESH:
            ret = k * DT
            break
    return {"base": base, "dbg": dbg, "tpeak": pk * DT, "auc": auc, "ret": float(ret),
            "curve": seg}


def size_bin(g):
    for i, (lo, hi) in enumerate(SIZE_BINS):
        if lo <= g < hi:
            return i
    return len(SIZE_BINS) - 1


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, name), dpi=130)
    plt.close(fig)


def fig_shape_by_size(curves_o, curves_u, fname):
    t = np.arange(WIN + 1) * DT
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharex=True)
    for b, ax in enumerate(axes.ravel()):
        lo, hi = SIZE_BINS[b]
        co = np.array(curves_o[b]) if curves_o[b] else np.zeros((1, WIN + 1))
        cu = np.array(curves_u[b]) if curves_u[b] else np.zeros((1, WIN + 1))
        for arr, c, lbl in ((co, C_OURS, "T1DMSIM"), (cu, C_UVA, "UVA/Padova")):
            m = arr.mean(0)
            sem = arr.std(0) / max(1, np.sqrt(len(arr)))
            ax.plot(t, m, color=c, lw=2, label=lbl)
            ax.fill_between(t, m - sem, m + sem, color=c, alpha=0.18)
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_title(f"{lo}–{hi if hi < 200 else '+'} g  (n={len(curves_o[b])})")
        ax.set_ylabel("BG − baseline (mg/dL)")
        ax.grid(alpha=0.25)
        if b == 0:
            ax.legend(loc="upper right")
    for ax in axes[1]:
        ax.set_xlabel("minutes after meal")
    fig.suptitle("Mean meal excursion by carbohydrate load (±SEM)")
    _save(fig, fname)


def fig_shape_normalized(curves_o, curves_u, fname):
    """Peak-normalised mean shape — amplitude removed, dynamics isolated."""
    t = np.arange(WIN + 1) * DT

    def norm_mean(bins):
        rows = []
        for b in bins:
            for c in b:
                pk = np.max(c)
                if pk > 10:
                    rows.append(np.asarray(c) / pk)
        return np.array(rows)

    fig, ax = plt.subplots(figsize=(11, 5.6))
    for bins, c, lbl in ((curves_o, C_OURS, "T1DMSIM"), (curves_u, C_UVA, "UVA/Padova")):
        arr = norm_mean(bins)
        m = arr.mean(0)
        sem = arr.std(0) / np.sqrt(len(arr))
        ax.plot(t, m, color=c, lw=2.2, label=f"{lbl}  (n={len(arr)})")
        ax.fill_between(t, m - sem, m + sem, color=c, alpha=0.18)
        ax.axvline(t[int(np.argmax(m))], color=c, ls=":", lw=1)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("minutes after meal")
    ax.set_ylabel("excursion / peak")
    ax.set_title("Peak-normalised excursion shape (dotted = mean time-to-peak)")
    ax.legend()
    ax.grid(alpha=0.25)
    _save(fig, fname)


def fig_peak_scatter(pairs, fname):
    do = np.array([p["o"]["dbg"] for p in pairs])
    du = np.array([p["u"]["dbg"] for p in pairs])
    g = np.array([p["g"] for p in pairs])
    fig, ax = plt.subplots(figsize=(8.4, 7))
    sc = ax.scatter(do, du, c=g, cmap="viridis", s=22, alpha=0.7)
    lim = [0, max(do.max(), du.max()) * 1.05]
    ax.plot(lim, lim, color="gray", ls="--", lw=1)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("T1DMSIM peak rise (mg/dL)")
    ax.set_ylabel("UVA/Padova peak rise (mg/dL)")
    ax.set_title("Paired meal peak rise")
    fig.colorbar(sc, ax=ax, label="meal carbs (g)")
    ax.grid(alpha=0.25)
    _save(fig, fname)


def fig_deviation_hist(pairs, fname):
    ddbg = np.array([p["o"]["dbg"] - p["u"]["dbg"] for p in pairs])
    dtp = np.array([p["o"]["tpeak"] - p["u"]["tpeak"] for p in pairs])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.hist(ddbg, bins=40, color="#555")
    ax1.axvline(float(np.median(ddbg)), color=C_OURS, lw=2,
                label=f"median {np.median(ddbg):+.0f}")
    ax1.set_xlabel("peak-rise difference  ours − UVA (mg/dL)")
    ax1.set_ylabel("meals")
    ax1.legend()
    ax1.grid(alpha=0.25)
    ax2.hist(dtp, bins=range(-WIN_MIN, WIN_MIN + 1, DT * 2), color="#555")
    ax2.axvline(float(np.median(dtp)), color=C_OURS, lw=2,
                label=f"median {np.median(dtp):+.0f} min")
    ax2.set_xlabel("time-to-peak difference  ours − UVA (min)")
    ax2.set_ylabel("meals")
    ax2.legend()
    ax2.grid(alpha=0.25)
    fig.suptitle("Per-meal excursion deviations")
    _save(fig, fname)


def fig_gallery(pairs_for_gallery, fname):
    n = len(pairs_for_gallery)
    fig, axes = plt.subplots(n, 1, figsize=(13, 2.3 * n), sharex=True)
    if n == 1:
        axes = [axes]
    warm = WARMUP_HOURS * 60 // DT
    span = int(60 * 60 / DT)
    for ax, p in zip(axes, pairs_for_gallery):
        a = p["bg_ours"][warm:warm + span]
        b = p["bg_uva"][warm:warm + span]
        t = np.arange(len(a)) * DT / 60.0
        ax.axhspan(70, 180, color="green", alpha=0.06)
        ax.plot(t, a, color=C_OURS, lw=1.3, label="T1DMSIM")
        ax.plot(t, b, color=C_UVA, lw=1.3, label="UVA/Padova (self-dosed)")
        for step, g in zip(p["meal_steps"], p["meal_g"]):
            if warm <= step < warm + span and g >= MEAL_MIN_G:
                ax.plot((step - warm) * DT / 60.0, 40, marker="^", color=C_MEAL, ms=6)
        ax.set_ylabel(f"seed {p['seed']}\n{p['uva_name']}", fontsize=8)
        ax.set_ylim(30, 360)
        ax.grid(alpha=0.2)
    axes[0].legend(loc="upper right", fontsize=8, ncol=2)
    axes[-1].set_xlabel("hours (post-warmup; ▲ = shared meal)")
    fig.suptitle("Paired BG traces — shared meals, each engine self-dosed")
    _save(fig, fname)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def med_iqr(v):
    v = np.asarray(v, dtype=float)
    if len(v) == 0:
        return {"median": float("nan"), "iqr_lo": float("nan"), "iqr_hi": float("nan"), "n": 0}
    return {"median": float(np.median(v)), "iqr_lo": float(np.percentile(v, 25)),
            "iqr_hi": float(np.percentile(v, 75)), "n": int(len(v))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--days", type=int, default=None)
    args = ap.parse_args()
    n_seeds = args.seeds or (5 if args.quick else N_SEEDS)
    days = args.days or (6 if args.quick else DAYS)
    os.makedirs(FIGS, exist_ok=True)

    print(f"Excursion comparison: {n_seeds} seeds x {days}d, {GROUP} cohort, "
          f"{MAX_WORKERS} workers")
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        pairs = list(ex.map(_worker, [(s, days) for s in range(n_seeds)]))
    pairs.sort(key=lambda p: p["seed"])

    warm = WARMUP_HOURS * 60 // DT
    matched, curves_o, curves_u = [], [[] for _ in SIZE_BINS], [[] for _ in SIZE_BINS]
    for p in pairs:
        n = len(p["bg_ours"])
        for s, g in detect_meals(p["meal_steps"], p["meal_g"], n):
            if s < warm:
                continue
            eo = excursion(p["bg_ours"], s)
            eu = excursion(p["bg_uva"], s)
            matched.append({"seed": p["seed"], "g": g, "o": eo, "u": eu})
            b = size_bin(g)
            curves_o[b].append(eo["curve"])
            curves_u[b].append(eu["curve"])
    print(f"  {len(matched)} isolated meal excursions extracted")

    # ---- figures ----
    fig_shape_by_size(curves_o, curves_u, "excursion_shape_by_size.png")
    fig_shape_normalized(curves_o, curves_u, "excursion_shape_normalized.png")
    fig_peak_scatter(matched, "excursion_peak_scatter.png")
    fig_deviation_hist(matched, "excursion_deviation_hist.png")
    gallery_seeds = list(range(0, n_seeds, max(1, n_seeds // 8)))[:8]
    fig_gallery([pairs[i] for i in gallery_seeds], "excursion_trace_gallery.png")

    # ---- stats ----
    def feat(side, k):
        return [m[side][k] for m in matched]

    per_bin = []
    for b, (lo, hi) in enumerate(SIZE_BINS):
        idx = [m for m in matched if size_bin(m["g"]) == b]
        per_bin.append({
            "bin": f"{lo}-{hi}", "n": len(idx),
            "peak_ours": med_iqr([m["o"]["dbg"] for m in idx]),
            "peak_uva": med_iqr([m["u"]["dbg"] for m in idx]),
            "tpeak_ours": med_iqr([m["o"]["tpeak"] for m in idx]),
            "tpeak_uva": med_iqr([m["u"]["tpeak"] for m in idx]),
        })

    stats = {
        "config": {
            "n_seeds": n_seeds, "days": days, "warmup_hours": WARMUP_HOURS, "dt_minutes": DT,
            "uva_group": GROUP, "n_excursions": len(matched),
            "meal_min_g": MEAL_MIN_G, "window_min": WIN_MIN,
            "uva_dosing": "own basal (u2ss) + grams/CR meal bolus + bounded CF correction",
            "shared_input": "meal schedule (time + grams) only; each engine self-doses",
        },
        "overall": {
            "peak_ours": med_iqr(feat("o", "dbg")), "peak_uva": med_iqr(feat("u", "dbg")),
            "tpeak_ours": med_iqr(feat("o", "tpeak")), "tpeak_uva": med_iqr(feat("u", "tpeak")),
            "auc_ours": med_iqr(feat("o", "auc")), "auc_uva": med_iqr(feat("u", "auc")),
            "return_ours": med_iqr(feat("o", "ret")), "return_uva": med_iqr(feat("u", "ret")),
            "peak_diff_ours_minus_uva": med_iqr([m["o"]["dbg"] - m["u"]["dbg"] for m in matched]),
            "tpeak_diff_ours_minus_uva": med_iqr([m["o"]["tpeak"] - m["u"]["tpeak"] for m in matched]),
        },
        "per_bin": per_bin,
    }
    with open(os.path.join(HERE, "excursions.json"), "w") as f:
        json.dump(stats, f, indent=2)

    write_report(stats)
    print(f"  wrote excursions.json, EXCURSIONS.md, 5 figures")


def write_report(stats):
    c = stats["config"]
    o = stats["overall"]

    def mi(d, fmt="{:.0f}"):
        return f"{fmt.format(d['median'])} ({fmt.format(d['iqr_lo'])}–{fmt.format(d['iqr_hi'])})"

    bin_rows = "\n".join(
        f"| {b['bin']} g | {b['n']} | {mi(b['peak_ours'])} | {mi(b['peak_uva'])} | "
        f"{mi(b['tpeak_ours'])} | {mi(b['tpeak_uva'])} |"
        for b in stats["per_bin"]
    )

    md = f"""# Meal excursions: T1DMSIM vs. a self-dosed UVA/Padova patient

Companion to [`README.md`](README.md), which replays T1DMSIM's insulin into
UVA/Padova verbatim: those boluses, sized by T1DMSIM's insulin-to-carb ratio,
over-cover the UVA virtual patients into sustained hypoglycaemia, so that
comparison is dominated by a gross level mismatch. Here the engines **share only
the meal schedule** and each doses insulin for its own physiology.

*Generated by `uva_padova/compare_excursions.py`. Do not edit by hand.*

## Method

For each of {c['n_seeds']} seeds, T1DMSIM generates a virtual patient and a
{c['days']}-day schedule. The food-meal times and grams are shared with a
paired UVA/Padova `{c['uva_group']}` patient; insulin is **not** shared.
T1DMSIM doses as it normally would (behavioural under-counting, missed boluses,
corrections, timing jitter and all); the UVA/Padova patient is dosed for its own
physiology — {c['uva_dosing']}.
Isolated post-meal excursions (≥{c['meal_min_g']:.0f} g, no other ≥{ISO_OTHER_G:.0f} g
meal within ±{ISO_PRE_MIN} min so the rise and peak are clean, a
{c['window_min']}-min window) are extracted relative to each engine's **own**
pre-meal baseline, so what is compared is the *response*, not the level.
{c['n_excursions']} excursions met the criteria.

## Excursion amplitude and timing (median, IQR)

| Quantity | T1DMSIM | UVA/Padova |
|---|---|---|
| Peak rise (mg/dL) | {mi(o['peak_ours'])} | {mi(o['peak_uva'])} |
| Time-to-peak (min) | {mi(o['tpeak_ours'])} | {mi(o['tpeak_uva'])} |
| 3-h excursion area (mg/dL·min) | {mi(o['auc_ours'])} | {mi(o['auc_uva'])} |
| Recovery to baseline (min) | {mi(o['return_ours'])} | {mi(o['return_uva'])} |

Per-meal paired differences (T1DMSIM − UVA/Padova): peak rise
**{o['peak_diff_ours_minus_uva']['median']:+.0f}** mg/dL
(IQR {o['peak_diff_ours_minus_uva']['iqr_lo']:+.0f}–{o['peak_diff_ours_minus_uva']['iqr_hi']:+.0f}),
time-to-peak **{o['tpeak_diff_ours_minus_uva']['median']:+.0f}** min
(IQR {o['tpeak_diff_ours_minus_uva']['iqr_lo']:+.0f}–{o['tpeak_diff_ours_minus_uva']['iqr_hi']:+.0f}).

### By carbohydrate load

| Meal size | n | T1DMSIM peak (mg/dL) | UVA/Padova peak (mg/dL) | T1DMSIM t-peak (min) | UVA/Padova t-peak (min) |
|---|---|---|---|---|---|
{bin_rows}

![Mean excursion by carbohydrate load](figures/excursion_shape_by_size.png)
![Paired peak rise](figures/excursion_peak_scatter.png)
![Per-meal deviations](figures/excursion_deviation_hist.png)

## Excursion shape (amplitude removed)

Each excursion is normalised by its own peak, leaving only the *shape* — how
fast glucose rises and how it returns.

![Peak-normalised excursion shape](figures/excursion_shape_normalized.png)

## Sample paired traces

Shared meals (▲), each engine self-dosed.

![Paired trace gallery](figures/excursion_trace_gallery.png)

## Reproducing

```bash
pip install --no-deps simglucose>=0.2.11
venv/bin/python uva_padova/compare_excursions.py            # full run
venv/bin/python uva_padova/compare_excursions.py --quick    # fast smoke run
```
"""
    with open(os.path.join(HERE, "EXCURSIONS.md"), "w") as f:
        f.write(md)


if __name__ == "__main__":
    main()
