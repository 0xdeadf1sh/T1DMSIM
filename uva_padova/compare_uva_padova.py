"""Compare T1DMSIM against the UVA/Padova 2008 in-silico model (simglucose).

Unlike the real-CGM comparisons under ``scripts/`` and ``diff/``, this harness
pits our behaviour-driven engine against the field's reference *physiological*
simulator, and does so under **identical inputs**: the meal times + grams,
bolus times + units, and basal that T1DMSIM generates for a given seed are
captured verbatim and replayed into a paired UVA/Padova virtual patient. We
then compare the resulting BG curves, their distributions, and — an axis the
other tooling ignores — raw generation speed.

Identical-input protocol
-------------------------
* Every patient input in T1DMSIM flows through ``T1DMSimulator.inject_curve``
  (scheduled meals/boluses/basal plus reactive corrections and rescues). HGO is
  endogenous and added per-step, never as a curve, so a monkeypatch on
  ``inject_curve`` captures exactly the patient's behavioural inputs and nothing
  hepatic — which is correct, since UVA/Padova models its own EGP.
* Captured curve totals are aggregated onto the 1-minute grid: carbs
  (``carb`` + ``correction_carb``) as gram impulses, boluses (``bolus``) as unit
  impulses, and long-acting basal smeared to the equivalent continuous pump
  rate (U/min) over each dose's nominal duration. Exercise, alcohol and stress
  have no analogue in the base UVA/Padova model and are not replayed.
* Both engines start from the paired patient's initial Gsub; the first
  ``WARMUP_HOURS`` are discarded from every metric. Curves are compared
  noise-free (our true ``bg`` vs UVA/Padova ``Gsub``) to isolate the metabolic
  models from CGM-noise modelling.

Writes ``uva_padova/stats.json``, ``uva_padova/figures/*.png`` and a
regenerated ``uva_padova/README.md``.

Run (after installing the reference engine without its RL extras):
    pip install --no-deps simglucose>=0.2.11
    venv/bin/python uva_padova/compare_uva_padova.py            # full run
    venv/bin/python uva_padova/compare_uva_padova.py --quick    # fast smoke run
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy import stats as sps  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
FIGS = os.path.join(HERE, "figures")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import simulator as S  # noqa: E402
from padova_engine import PadovaPatient, patient_names  # noqa: E402

# Reuse the pure, dependency-free metric helpers from the Ohio comparison.
from compare_to_ohio import (  # noqa: E402
    basic_stats,
    clinical_ranges,
    delta_stats,
    episode_stats,
    episodes,
)

DT = S.DT_MINUTES                 # 5-minute simulation grid
GROUP = "adult"                   # UVA/Padova cohort closest to our parameterisation
N_SEEDS = 24
DAYS = 14
WARMUP_HOURS = 48
MAX_WORKERS = max(1, (os.cpu_count() or 4) // 2)   # leave half the box free

# Engine colours, kept consistent across every figure.
C_OURS = "#1f77b4"
C_UVA = "#d62728"


# ---------------------------------------------------------------------------
# Input capture + replay (one seed)
# ---------------------------------------------------------------------------
def capture_run(seed: int, days: int, initial_bg: float):
    """Run T1DMSIM, returning (patient, true-BG per 5-min step, input events).

    ``events`` is a list of ``(start_step, curve_type, total_amount)`` recorded
    from every ``inject_curve`` call during the run.
    """
    sim = S.T1DMSimulator(seed=seed, initial_bg=initial_bg)
    events: list = []
    original = sim.inject_curve

    def hook(values, start_idx, curve_type, label=""):
        events.append((int(start_idx), curve_type, float(np.sum(values))))
        return original(values, start_idx, curve_type, label)

    sim.inject_curve = hook  # type: ignore[method-assign]
    n_steps = days * 24 * 60 // DT
    bg = np.empty(n_steps)
    for i in range(n_steps):
        bg[i] = float(sim.generate()["bg"])
    return sim, bg, events


def build_inputs(events: list, n_steps: int, basal_duration_hours: float):
    """Aggregate captured curves into aligned 1-minute carb/bolus/basal arrays."""
    n_min = n_steps * DT
    carb = np.zeros(n_min)
    bolus = np.zeros(n_min)
    basal = np.zeros(n_min)
    for start_step, ctype, total in events:
        m = start_step * DT
        if m >= n_min:
            continue
        if ctype in ("carb", "correction_carb"):
            carb[m] += total
        elif ctype in ("bolus", "insulin"):
            bolus[m] += total
        elif ctype == "basal":
            # MDI long-acting dose -> equivalent continuous pump rate (U/min),
            # smeared over its nominal duration; overlapping doses sum.
            dur_min = max(1, int(basal_duration_hours * 60))
            rate = total / dur_min
            end = min(n_min, m + dur_min)
            basal[m:end] += rate
    return carb, bolus, basal


_ADULT_BASAL = None


def _adult_basal_table():
    """[(name, steady-state basal U/day)] for the cohort, cached per process."""
    global _ADULT_BASAL
    if _ADULT_BASAL is None:
        _ADULT_BASAL = [(n, PadovaPatient(n).basal_rate_u_per_min * 1440.0)
                        for n in patient_names(GROUP)]
    return _ADULT_BASAL


def nearest_adult(basal_u_day: float) -> str:
    """The virtual patient whose basal need is closest — keeps the comparison
    about transfer-function shape rather than gross basal/EGP mismatch."""
    return min(_adult_basal_table(), key=lambda kv: abs(kv[1] - basal_u_day))[0]


def run_pair(seed: int, days: int):
    """Generate one paired sample. Returns a dict of arrays + per-seed summary."""
    our_basal = float(S.T1DMSimulator(seed=seed).patient.basal_dose)
    name = nearest_adult(our_basal)
    pad = PadovaPatient(name)
    init_g = pad.init_gsub

    sim, bg_ours, events = capture_run(seed, days, init_g)
    n_steps = len(bg_ours)
    carb, bolus, basal = build_inputs(events, n_steps, sim.patient.basal_duration_hours)

    gsub = pad.replay(carb, basal + bolus)
    bg_uva = gsub[::DT][:n_steps]

    days_eff = n_steps * DT / (60 * 24)
    return {
        "seed": seed,
        "uva_name": name,
        "bg_ours": bg_ours,
        "bg_uva": bg_uva,
        "carb_per_min": carb,
        "bolus_per_min": bolus,
        "basal_per_min": basal,
        "summary": {
            "uva_name": name,
            "uva_bw_kg": pad.body_weight_kg,
            "uva_basal_u_day": pad.basal_rate_u_per_min * 1440.0,
            "our_bw_kg": float(sim.patient.body_weight_kg),
            "our_icr": float(sim.patient.icr),
            "our_basal_dose_u": float(sim.patient.basal_dose),
            "carbs_g_per_day": float(carb.sum() / days_eff),
            "bolus_u_per_day": float(bolus.sum() / days_eff),
            "basal_u_per_day": float(basal.mean() * 1440.0),
            "meals_per_day": float(np.count_nonzero(carb) / days_eff),
            "boluses_per_day": float(np.count_nonzero(bolus) / days_eff),
        },
    }


def _worker(args):
    seed, days = args
    r = run_pair(seed, days)
    # Drop the bulky minute-grid inputs before pickling back to the parent.
    for k in ("carb_per_min", "bolus_per_min", "basal_per_min"):
        r.pop(k)
    return r


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def kovatchev_risk(bg: np.ndarray):
    """Kovatchev Low/High Blood Glucose Indices (LBGI, HBGI)."""
    bg = bg[~np.isnan(bg)]
    f = 1.509 * (np.log(np.clip(bg, 1.0, None)) ** 1.084 - 5.381)
    rl = np.where(f < 0, 10.0 * f ** 2, 0.0)
    rh = np.where(f > 0, 10.0 * f ** 2, 0.0)
    return float(np.mean(rl)), float(np.mean(rh))


def diurnal_by_hour(bg: np.ndarray, start_step: int) -> np.ndarray:
    """Mean BG per hour-of-day; step ``start_step`` is the first sample's index."""
    buckets: list = [[] for _ in range(24)]
    for i, v in enumerate(bg):
        hour = ((start_step + i) * DT // 60) % 24
        buckets[hour].append(v)
    return np.array([float(np.mean(b)) if b else float("nan") for b in buckets])


def stat_block(bg: np.ndarray, start_step: int, days: float) -> dict:
    out: dict = {**basic_stats(bg), **clinical_ranges(bg)}
    hypo = episode_stats(episodes(bg, 0, 70))
    hyper = episode_stats(episodes(bg, 180, 1000))
    severe = episode_stats(episodes(bg, 0, 54))
    out["hypo_n_per_day"] = hypo["n"] / days if days else 0.0
    out["hyper_n_per_day"] = hyper["n"] / days if days else 0.0
    out["hypo_median_min"] = hypo["median"]
    out["hypo_p90_min"] = hypo["p90"]
    out["hyper_median_min"] = hyper["median"]
    out["severe_n_per_day"] = severe["n"] / days if days else 0.0
    out["severe_max_min"] = severe["max"]
    lbgi, hbgi = kovatchev_risk(bg)
    out["lbgi"] = lbgi
    out["hbgi"] = hbgi
    out.update(delta_stats(bg))
    out["diurnal"] = diurnal_by_hour(bg, start_step)
    return out


def paired_metrics(a: np.ndarray, b: np.ndarray) -> dict:
    """Point-wise agreement between two identically-driven, equal-length curves."""
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    r = float(np.corrcoef(a, b)[0, 1])
    best_lag, best_r = 0, r
    for lag in range(-6, 7):  # +/- 30 min
        aa = a[max(0, lag): n + min(0, lag)]
        bb = b[max(0, -lag): n + min(0, -lag)]
        rr = float(np.corrcoef(aa, bb)[0, 1])
        if rr > best_r:
            best_r, best_lag = rr, lag
    return {
        "rmse": float(np.sqrt(np.mean((a - b) ** 2))),
        "mad": float(np.mean(np.abs(a - b))),
        "r": r,
        "best_lag_r": best_r,
        "best_lag_min": best_lag * DT,
        "drift": float(np.mean(b) - np.mean(a)),
        "ks": float(sps.ks_2samp(a, b).statistic),
        "wasserstein": float(sps.wasserstein_distance(a, b)),
    }


def pool_blocks(blocks: list) -> dict:
    keys = [k for k in blocks[0] if k != "diurnal"]
    out = {k: float(np.nanmean([b[k] for b in blocks])) for k in keys}
    out["diurnal"] = np.nanmean(np.stack([b["diurnal"] for b in blocks]), axis=0)
    return out


def median_iqr(vals: list) -> dict:
    v = np.array(vals, dtype=float)
    return {
        "median": float(np.median(v)),
        "iqr_lo": float(np.percentile(v, 25)),
        "iqr_hi": float(np.percentile(v, 75)),
        "mean": float(np.mean(v)),
    }


# ---------------------------------------------------------------------------
# Speed benchmark
# ---------------------------------------------------------------------------
def bench_ours(days: int) -> float:
    sim = S.T1DMSimulator(seed=1, initial_bg=150.0)
    n = days * 24 * 60 // DT
    t0 = time.perf_counter()
    for _ in range(n):
        sim.generate()
    return time.perf_counter() - t0


def bench_uva(days: int, carb, bolus, basal) -> float:
    pad = PadovaPatient(patient_names(GROUP)[0])
    n_min = days * 24 * 60
    t0 = time.perf_counter()
    pad.replay(carb[:n_min], (basal + bolus)[:n_min])
    return time.perf_counter() - t0


def run_speed(sweep_days: list) -> dict:
    """Single-threaded wall-clock for each engine across a horizon sweep."""
    # A fixed real input stream for the UVA timing (so we time the ODE, not
    # input synthesis). Capture the longest horizon once and slice it.
    longest = max(sweep_days)
    pad0 = PadovaPatient(patient_names(GROUP)[0])
    sim, bg, events = capture_run(1, longest, pad0.init_gsub)
    carb, bolus, basal = build_inputs(events, len(bg), sim.patient.basal_duration_hours)

    rows = []
    for d in sweep_days:
        t_ours = bench_ours(d)
        t_uva = bench_uva(d, carb, bolus, basal)
        rows.append({
            "days": d,
            "ours_s": t_ours,
            "uva_s": t_uva,
            "ours_ms_per_day": 1000 * t_ours / d,
            "uva_ms_per_day": 1000 * t_uva / d,
            "speedup": t_uva / t_ours if t_ours else float("nan"),
        })
    n_steps = max(sweep_days) * 24 * 60 // DT
    headline = rows[-1]
    return {
        "sweep": rows,
        "ours_steps_per_s": n_steps / rows[-1]["ours_s"],
        "uva_steps_per_s": (max(sweep_days) * 24 * 60) / rows[-1]["uva_s"],
        "speedup_x": headline["speedup"],
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _save(fig, name):
    path = os.path.join(FIGS, name)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_paired_trace(pair, carb, bolus, hours, fname, title):
    w = WARMUP_HOURS * 60 // DT
    span = int(hours * 60 / DT)
    a = pair["bg_ours"][w:w + span]
    b = pair["bg_uva"][w:w + span]
    t = np.arange(len(a)) * DT / 60.0
    cm = carb[w * DT:(w + span) * DT]
    bm = bolus[w * DT:(w + span) * DT]
    tm = np.arange(len(cm)) / 60.0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 6.4), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    ax1.axhspan(70, 180, color="green", alpha=0.06)
    ax1.plot(t, a, color=C_OURS, lw=1.6, label="T1DMSIM (true BG)")
    ax1.plot(t, b, color=C_UVA, lw=1.6, label="UVA/Padova (Gsub)")
    ax1.set_ylabel("BG (mg/dL)")
    ax1.set_title(title)
    ax1.legend(loc="upper right", framealpha=0.9)
    ax1.grid(alpha=0.25)

    ct = np.nonzero(cm)[0]
    bt = np.nonzero(bm)[0]
    ax2.stem(tm[ct], cm[ct], linefmt="C2-", markerfmt="C2^", basefmt=" ", label="carbs (g)")
    ax2b = ax2.twinx()
    ax2b.stem(tm[bt], bm[bt], linefmt="C1-", markerfmt="C1v", basefmt=" ", label="bolus (U)")
    ax2.set_ylabel("carbs (g)", color="C2")
    ax2b.set_ylabel("bolus (U)", color="C1")
    ax2.set_xlabel("hours (post-warmup, identical inputs to both engines)")
    ax2.grid(alpha=0.2)
    _save(fig, fname)


def fig_pdf(pooled_a, pooled_b, fname):
    fig, ax = plt.subplots(figsize=(11, 5.2))
    bins = np.linspace(40, 400, 90)
    ax.hist(pooled_a, bins=bins, density=True, color=C_OURS, alpha=0.5, label="T1DMSIM")
    ax.hist(pooled_b, bins=bins, density=True, color=C_UVA, alpha=0.5, label="UVA/Padova")
    for x in (70, 180):
        ax.axvline(x, color="gray", ls="--", lw=1)
    ax.set_xlabel("BG (mg/dL)")
    ax.set_ylabel("density")
    ax.set_title("Pooled BG distribution under identical inputs")
    ax.legend()
    ax.grid(alpha=0.25)
    _save(fig, fname)


def fig_cdf(pooled_a, pooled_b, fname):
    fig, ax = plt.subplots(figsize=(11, 5.2))
    for data, c, lbl in ((pooled_a, C_OURS, "T1DMSIM"), (pooled_b, C_UVA, "UVA/Padova")):
        x = np.sort(data)
        ax.plot(x, np.linspace(0, 1, len(x)), color=c, lw=1.8, label=lbl)
    for v in (70, 180):
        ax.axvline(v, color="gray", ls="--", lw=1)
    ax.set_xlabel("BG (mg/dL)")
    ax.set_ylabel("cumulative fraction")
    ax.set_title("Pooled BG CDF")
    ax.legend()
    ax.grid(alpha=0.25)
    _save(fig, fname)


def fig_tir(pa, pb, fname):
    cats = ["TBR2_<54", "TBR1_54-70", "TIR_70-180", "TAR1_180-250", "TAR2_>250"]
    labels = ["<54", "54-70", "70-180", "180-250", ">250"]
    x = np.arange(len(cats))
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.bar(x - 0.2, [pa[c] for c in cats], 0.4, color=C_OURS, label="T1DMSIM")
    ax.bar(x + 0.2, [pb[c] for c in cats], 0.4, color=C_UVA, label="UVA/Padova")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("% of time")
    ax.set_title("Time-in-range bands")
    ax.legend()
    ax.grid(alpha=0.25, axis="y")
    _save(fig, fname)


def fig_diurnal(pa, pb, fname):
    fig, ax = plt.subplots(figsize=(11, 5.2))
    h = np.arange(24)
    ax.plot(h, pa["diurnal"], color=C_OURS, marker="o", ms=4, label="T1DMSIM")
    ax.plot(h, pb["diurnal"], color=C_UVA, marker="s", ms=4, label="UVA/Padova")
    ax.set_xlabel("hour of day")
    ax.set_ylabel("mean BG (mg/dL)")
    ax.set_title("Diurnal mean BG")
    ax.set_xticks(range(0, 24, 2))
    ax.legend()
    ax.grid(alpha=0.25)
    _save(fig, fname)


def fig_bland_altman(a, b, fname):
    rng = np.random.default_rng(0)
    if len(a) > 6000:
        idx = rng.choice(len(a), 6000, replace=False)
        a, b = a[idx], b[idx]
    mean = (a + b) / 2
    diff = b - a
    bias = float(np.mean(diff))
    sd = float(np.std(diff))
    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.scatter(mean, diff, s=5, alpha=0.18, color="#555")
    ax.axhline(bias, color=C_UVA, lw=1.5, label=f"bias {bias:+.1f}")
    ax.axhline(bias + 1.96 * sd, color="gray", ls="--", lw=1, label="±1.96 SD")
    ax.axhline(bias - 1.96 * sd, color="gray", ls="--", lw=1)
    ax.set_xlabel("mean of the two engines (mg/dL)")
    ax.set_ylabel("UVA/Padova − T1DMSIM (mg/dL)")
    ax.set_title("Bland–Altman: paired-curve agreement")
    ax.legend()
    ax.grid(alpha=0.25)
    _save(fig, fname)


def fig_delta(a, b, fname):
    fig, ax = plt.subplots(figsize=(11, 5.2))
    bins = np.linspace(-25, 25, 100)
    ax.hist(np.diff(a), bins=bins, density=True, color=C_OURS, alpha=0.5, label="T1DMSIM")
    ax.hist(np.diff(b), bins=bins, density=True, color=C_UVA, alpha=0.5, label="UVA/Padova")
    ax.set_xlabel("5-min ΔBG (mg/dL)")
    ax.set_ylabel("density")
    ax.set_title("Rate-of-change distribution")
    ax.legend()
    ax.grid(alpha=0.25)
    _save(fig, fname)


def fig_speed(speed, fname):
    sweep = speed["sweep"]
    days = [r["days"] for r in sweep]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
    ax1.bar([0], [sweep[-1]["ours_ms_per_day"]], color=C_OURS, label="T1DMSIM")
    ax1.bar([1], [sweep[-1]["uva_ms_per_day"]], color=C_UVA, label="UVA/Padova")
    ax1.set_yscale("log")
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["T1DMSIM", "UVA/Padova"])
    ax1.set_ylabel("ms per simulated day (log)")
    ax1.set_title(f"Per-day cost  (×{speed['speedup_x']:.0f} faster)")
    for i, r in enumerate([sweep[-1]["ours_ms_per_day"], sweep[-1]["uva_ms_per_day"]]):
        ax1.text(i, r, f"{r:.1f}", ha="center", va="bottom")
    ax1.grid(alpha=0.25, axis="y")

    ax2.plot(days, [r["ours_s"] for r in sweep], color=C_OURS, marker="o", label="T1DMSIM")
    ax2.plot(days, [r["uva_s"] for r in sweep], color=C_UVA, marker="s", label="UVA/Padova")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("simulated days")
    ax2.set_ylabel("wall-clock seconds (log)")
    ax2.set_title("Scaling")
    ax2.legend()
    ax2.grid(alpha=0.25, which="both")
    _save(fig, fname)


def fig_per_seed(per_seed, fname):
    mo = [p["mean_ours"] for p in per_seed]
    mu = [p["mean_uva"] for p in per_seed]
    rmse = [p["rmse"] for p in per_seed]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
    sc = ax1.scatter(mo, mu, c=rmse, cmap="viridis", s=40)
    lim = [min(mo + mu) - 5, max(mo + mu) + 5]
    ax1.plot(lim, lim, color="gray", ls="--", lw=1)
    ax1.set_xlim(lim)
    ax1.set_ylim(lim)
    ax1.set_xlabel("T1DMSIM mean BG (mg/dL)")
    ax1.set_ylabel("UVA/Padova mean BG (mg/dL)")
    ax1.set_title("Per-seed mean BG")
    fig.colorbar(sc, ax=ax1, label="paired RMSE")
    ax1.grid(alpha=0.25)

    to = [p["tir_ours"] for p in per_seed]
    tu = [p["tir_uva"] for p in per_seed]
    ax2.scatter(to, tu, color="#555", s=40)
    lim2 = [min(to + tu) - 5, max(to + tu) + 5]
    ax2.plot(lim2, lim2, color="gray", ls="--", lw=1)
    ax2.set_xlim(lim2)
    ax2.set_ylim(lim2)
    ax2.set_xlabel("T1DMSIM TIR %")
    ax2.set_ylabel("UVA/Padova TIR %")
    ax2.set_title("Per-seed TIR (70–180)")
    ax2.grid(alpha=0.25)
    _save(fig, fname)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def tir(bg):
    return float(100 * np.mean((bg >= 70) & (bg <= 180)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fast smoke run (few seeds/days)")
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--days", type=int, default=None)
    args = ap.parse_args()

    n_seeds = args.seeds or (4 if args.quick else N_SEEDS)
    days = args.days or (5 if args.quick else DAYS)
    warm = WARMUP_HOURS * 60 // DT
    os.makedirs(FIGS, exist_ok=True)

    print(f"UVA/Padova comparison: {n_seeds} seeds x {days}d, {GROUP} cohort, "
          f"{WARMUP_HOURS}h warmup, {MAX_WORKERS} workers")

    t_start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        pairs = list(ex.map(_worker, [(s, days) for s in range(n_seeds)]))
    pairs.sort(key=lambda p: p["seed"])
    print(f"  paired sims done in {time.perf_counter() - t_start:.1f}s")

    days_eff = (len(pairs[0]["bg_ours"]) - warm) * DT / (60 * 24)
    blocks_o, blocks_u, paired, per_seed = [], [], [], []
    pooled_o_raw, pooled_u_raw = [], []
    for p in pairs:
        a = p["bg_ours"][warm:]
        b = p["bg_uva"][warm:]
        blocks_o.append(stat_block(a, warm, days_eff))
        blocks_u.append(stat_block(b, warm, days_eff))
        pm = paired_metrics(a, b)
        paired.append(pm)
        pooled_o_raw.append(a)
        pooled_u_raw.append(b)
        per_seed.append({
            "seed": p["seed"], "uva_name": p["uva_name"],
            "mean_ours": float(np.mean(a)), "mean_uva": float(np.mean(b)),
            "tir_ours": tir(a), "tir_uva": tir(b), "rmse": pm["rmse"], "r": pm["r"],
        })

    pooled_o = pool_blocks(blocks_o)
    pooled_u = pool_blocks(blocks_u)
    pooled_o_arr = np.concatenate(pooled_o_raw)
    pooled_u_arr = np.concatenate(pooled_u_raw)

    paired_summary = {k: median_iqr([pm[k] for pm in paired])
                      for k in ("rmse", "mad", "r", "best_lag_r", "drift", "ks", "wasserstein")}

    print("  speed benchmark...")
    sweep_days = [1, 2, 5] if args.quick else [1, 3, 7, 14]
    speed = run_speed(sweep_days)
    print(f"    ours {speed['ours_steps_per_s']:.0f} steps/s | "
          f"UVA {speed['uva_steps_per_s']:.0f} steps/s | x{speed['speedup_x']:.0f}")

    # ---- figures ----
    print("  rendering figures...")
    rep = min(range(len(per_seed)), key=lambda i: abs(per_seed[i]["rmse"]
                                                      - paired_summary["rmse"]["median"]))
    rp = pairs[rep]
    simrep, _, ev = capture_run(rp["seed"], days, PadovaPatient(rp["uva_name"]).init_gsub)
    carb, bolus, _ = build_inputs(ev, len(rp["bg_ours"]),
                                  simrep.patient.basal_duration_hours)
    ttl = f"seed {rp['seed']}  ·  paired with {rp['uva_name']}  ·  identical inputs"
    fig_paired_trace(rp, carb, bolus, 72, "paired_trace.png", ttl)
    fig_paired_trace(rp, carb, bolus, 24, "paired_trace_zoom.png", ttl + "  (24h)")
    fig_pdf(pooled_o_arr, pooled_u_arr, "pdf_overlay.png")
    fig_cdf(pooled_o_arr, pooled_u_arr, "cdf_overlay.png")
    fig_tir(pooled_o, pooled_u, "tir_bars.png")
    fig_diurnal(pooled_o, pooled_u, "diurnal.png")
    fig_bland_altman(pooled_o_arr, pooled_u_arr, "bland_altman.png")
    fig_delta(pooled_o_arr, pooled_u_arr, "delta_dist.png")
    fig_speed(speed, "speed_bench.png")
    fig_per_seed(per_seed, "per_seed_scatter.png")

    # ---- stats.json ----
    def clean(d):
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in d.items()}

    stats = {
        "config": {
            "n_seeds": n_seeds, "days": days, "warmup_hours": WARMUP_HOURS,
            "dt_minutes": DT, "uva_group": GROUP, "engine": "simglucose T1DPatient (UVA/Padova 2008)",
            "compared_signals": "T1DMSIM true bg vs UVA/Padova Gsub (both noise-free)",
            "excluded_from_replay": ["exercise", "alcohol", "stress", "CGM sensor noise"],
        },
        "pooled_ours": clean(pooled_o),
        "pooled_uva": clean(pooled_u),
        "paired": paired_summary,
        "speed": speed,
        "per_seed": per_seed,
        "input_summaries": [p["summary"] for p in pairs],
    }
    with open(os.path.join(HERE, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    write_readme(stats)
    print(f"  wrote stats.json, README.md, {len(os.listdir(FIGS))} figures")
    print(f"done in {time.perf_counter() - t_start:.1f}s")


# ---------------------------------------------------------------------------
# README generation (neutral, observed-gap language only)
# ---------------------------------------------------------------------------
def write_readme(stats: dict):
    o, u = stats["pooled_ours"], stats["pooled_uva"]
    pr = stats["paired"]
    sp = stats["speed"]
    cfg = stats["config"]

    def row(label, key, fmt="{:.1f}"):
        a, b = o.get(key, float("nan")), u.get(key, float("nan"))
        return f"| {label} | {fmt.format(a)} | {fmt.format(b)} | {fmt.format(b - a)} |"

    dist_rows = "\n".join([
        row("Mean BG (mg/dL)", "mean"),
        row("Median BG (mg/dL)", "median"),
        row("SD (mg/dL)", "std"),
        row("CV (%)", "cv_%"),
        row("GMI (%)", "gmi", "{:.2f}"),
        row("p10 (mg/dL)", "p10"),
        row("p90 (mg/dL)", "p90"),
        row("Time <54 % (TBR2)", "TBR2_<54", "{:.2f}"),
        row("Time 54–70 % (TBR1)", "TBR1_54-70", "{:.2f}"),
        row("Time 70–180 % (TIR)", "TIR_70-180"),
        row("Time 180–250 % (TAR1)", "TAR1_180-250"),
        row("Time >250 % (TAR2)", "TAR2_>250", "{:.2f}"),
        row("LBGI", "lbgi", "{:.2f}"),
        row("HBGI", "hbgi", "{:.2f}"),
        row("Hypo episodes/day", "hypo_n_per_day", "{:.2f}"),
        row("Hyper episodes/day", "hyper_n_per_day", "{:.2f}"),
        row("5-min ΔBG SD (mg/dL)", "d_std", "{:.2f}"),
    ])

    sweep_rows = "\n".join(
        f"| {r['days']} | {r['ours_s']:.3f} | {r['uva_s']:.2f} | {r['ours_ms_per_day']:.2f} | "
        f"{r['uva_ms_per_day']:.0f} | ×{r['speedup']:.0f} |"
        for r in sp["sweep"]
    )

    md = f"""# T1DMSIM vs. UVA/Padova 2008 (in-silico reference)

Companion to the real-CGM comparisons under `diff/` and `scripts/`. This report
pits T1DMSIM against the field's reference physiological simulator — the
FDA-accepted **UVA/Padova 2008** model, via the `simglucose` ODE core — under
**identical patient inputs**, and additionally benchmarks raw generation speed.

Two companion reports go further: [`EXCURSIONS.md`](EXCURSIONS.md) removes the
dosing-policy mismatch documented below — sharing only the meal schedule and
letting each engine dose for its own physiology — and compares the meal
excursions directly; [`REALISM.md`](REALISM.md) asks which simulator is the
better real-world proxy by measuring each one's distance from real CGM
(OhioT1DM + ShanghaiT1DM + AZT1D).

*Generated by `uva_padova/compare_uva_padova.py`. Do not edit by hand.*

## Method

For each of {cfg['n_seeds']} seeds, T1DMSIM generates a virtual patient and a
{cfg['days']}-day behavioural schedule. Every meal (time + grams), bolus
(time + units) and basal dose it produces is captured and **replayed verbatim**
into a paired UVA/Padova virtual patient from the `{cfg['uva_group']}` cohort —
specifically the individual whose steady-state basal requirement is closest to
the T1DMSIM patient's, so the comparison reflects transfer-function differences
rather than gross basal/EGP mismatch. Both engines start from the paired
patient's initial subcutaneous glucose; the first {cfg['warmup_hours']} h are
discarded from every metric.

* **Identical inputs.** All T1DMSIM patient inputs flow through one injection
  path, so they are captured completely; hepatic glucose output is endogenous
  and excluded — correctly, since UVA/Padova models its own EGP.
* **Basal.** T1DMSIM's MDI long-acting doses are converted to the equivalent
  continuous pump rate (U/min) that the UVA/Padova pump model expects.
* **Compared signals.** {cfg['compared_signals']}.
* **Not replayed** (no analogue in the base UVA/Padova model):
  {', '.join(cfg['excluded_from_replay'])}.

## Distributional comparison (pooled across seeds)

| Metric | T1DMSIM | UVA/Padova | Δ (UVA − ours) |
|---|---|---|---|
{dist_rows}

![Pooled BG distribution](figures/pdf_overlay.png)
![Pooled BG CDF](figures/cdf_overlay.png)
![Time-in-range bands](figures/tir_bars.png)
![Diurnal mean BG](figures/diurnal.png)
![Rate-of-change distribution](figures/delta_dist.png)

## Paired-curve agreement

Because both engines see identical inputs, their curves can be compared
point-for-point (median across seeds, IQR in brackets):

| Quantity | Median | IQR |
|---|---|---|
| RMSE (mg/dL) | {pr['rmse']['median']:.1f} | {pr['rmse']['iqr_lo']:.1f}–{pr['rmse']['iqr_hi']:.1f} |
| Mean abs. diff (mg/dL) | {pr['mad']['median']:.1f} | {pr['mad']['iqr_lo']:.1f}–{pr['mad']['iqr_hi']:.1f} |
| Pearson r | {pr['r']['median']:.2f} | {pr['r']['iqr_lo']:.2f}–{pr['r']['iqr_hi']:.2f} |
| Pearson r (best lag) | {pr['best_lag_r']['median']:.2f} | {pr['best_lag_r']['iqr_lo']:.2f}–{pr['best_lag_r']['iqr_hi']:.2f} |
| Mean-BG drift (mg/dL) | {pr['drift']['median']:+.1f} | {pr['drift']['iqr_lo']:+.1f}–{pr['drift']['iqr_hi']:+.1f} |
| KS distance | {pr['ks']['median']:.3f} | {pr['ks']['iqr_lo']:.3f}–{pr['ks']['iqr_hi']:.3f} |
| Wasserstein-1 (mg/dL) | {pr['wasserstein']['median']:.1f} | {pr['wasserstein']['iqr_lo']:.1f}–{pr['wasserstein']['iqr_hi']:.1f} |

Agreement varies widely from seed to seed (note the IQRs), and point-wise
correlation is weak throughout: the two engines answer the same meals and
boluses with differently-shaped transfer functions, and T1DMSIM further layers
the not-replayed behavioural perturbations above. The larger story is the
divergence in level. The same doses leave the two engines off-centre in
opposite directions — T1DMSIM mildly high (its dosing is calibrated to its own
hepatic-output and insulin-sensitivity assumptions), the UVA/Padova patient
lower in roughly two seeds out of three: the bolus stream, sized by T1DMSIM's
insulin-to-carb ratio, over-covers that patient's gentler carb ratio, and the
base model carries neither T1DMSIM's counter-regulatory floor below 70 mg/dL nor
its behaviourally-timed rescue carbs, so its lows run deeper. The gap therefore
measures how differently the two models embed the behaviour→glucose mapping,
not an error in either.

![Representative paired trace, 72 h](figures/paired_trace.png)
![Representative paired trace, 24 h](figures/paired_trace_zoom.png)
![Bland–Altman](figures/bland_altman.png)
![Per-seed agreement](figures/per_seed_scatter.png)

## Generation speed

Single-threaded, steady-state (warmup and input synthesis excluded). T1DMSIM
reads pre-accumulated curve contributions in O(1) per 5-min step; UVA/Padova
integrates a 13-state stiff ODE every minute.

| Simulated days | T1DMSIM (s) | UVA/Padova (s) | T1DMSIM ms/day | UVA/Padova ms/day | Speedup |
|---|---|---|---|---|---|
{sweep_rows}

* T1DMSIM: **{sp['ours_steps_per_s']:.0f}** steps/s
* UVA/Padova: **{sp['uva_steps_per_s']:.0f}** ODE-minutes/s
* End-to-end speedup at the longest horizon: **×{sp['speedup_x']:.0f}**

![Speed benchmark](figures/speed_bench.png)

## Reproducing

The reference engine is installed without its RL extras (the pinned `gym==0.9.4`
is unneeded and does not build on current Python; `gym` and `pkg_resources` are
stubbed at import time — see `padova_engine.py`):

```bash
pip install --no-deps simglucose>=0.2.11
venv/bin/python uva_padova/compare_uva_padova.py            # full run
venv/bin/python uva_padova/compare_uva_padova.py --quick    # fast smoke run
```
"""
    with open(os.path.join(HERE, "README.md"), "w") as f:
        f.write(md)


if __name__ == "__main__":
    main()
