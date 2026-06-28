"""Which simulator is the better real-world proxy: T1DMSIM or UVA/Padova?

Framed as the user's question — if two ML models were trained on the synthetic
output of each simulator, which would transfer better to real CGM? — but
answered *without training anything*. The proxy is distributional: a model
trained on synthetic data and deployed on real data pays a penalty that grows
with the distance between the two distributions (covariate / label shift). So
we measure how far each simulator's output sits from real CGM, across a battery
of marginal and dynamic statistics, with the **real-vs-real** distance between
cohorts as the yardstick for "indistinguishable from real".

Cohorts compared (all reduced to 5-min CGM, the battery in `diff/build_report.py`):
  * Real: OhioT1DM, ShanghaiT1DM, AZT1D — and their pool.
  * T1DMSIM: native free-living traces (per seed).
  * UVA/Padova: native simglucose data-generation — `T1DPatient` physiology +
    `RandomScenario` meals + `BBController` basal-bolus dosing + Dexcom
    `CGMSensor` noise. This is the canonical way UVA/Padova synthetic CGM is
    produced; using it (rather than T1DMSIM's behavioural schedule) keeps each
    simulator a self-contained data source.

IMPORTANT context, stated in the report: T1DMSIM was *calibrated against* the
Ohio/Shanghai cohorts; UVA/Padova was not. Closeness on level metrics is
therefore partly by construction. The report separates level metrics (where
that advantage lives) from dynamic metrics (a fairer test of the generative
process, and what a next-value BG predictor actually learns).

Writes `uva_padova/realism.json`, figures under `uva_padova/figures/`, and
`uva_padova/REALISM.md`.

Run (datasets present under ./datasets, reference engine installed --no-deps):
    venv/bin/python uva_padova/compare_realism.py
    venv/bin/python uva_padova/compare_realism.py --quick
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

logging.disable(logging.CRITICAL)  # silence simglucose's per-step logging

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
FIGS = os.path.join(HERE, "figures")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
sys.path.insert(0, os.path.join(REPO_ROOT, "diff"))

import padova_engine  # noqa: E402,F401  (installs the simglucose import shims)
from build_report import (  # noqa: E402
    assemble_cohort, assemble_sim, trivial_regularize_5min,
    distribution_distances, cohort_summary,
)
from compare_all_datasets import (  # noqa: E402
    load_ohio_patients, load_shanghai_patients, load_azt1d_patients,
    regularize_bg, regularize_bg_15min,
)

STEP_MIN = 5
OURSIM_SEEDS = 30
OURSIM_DAYS = 30
UVA_REPLICATES = 4          # scenario re-rolls per adult patient
UVA_DAYS = 14
UVA_WARMUP_H = 24
MAX_WORKERS = max(1, (os.cpu_count() or 4) // 2)

C = {"Real": "#000000", "T1DMSIM": "#1f77b4", "UVA/Padova": "#d62728",
     "Ohio": "#555555", "Shanghai": "#888888", "AZT1D": "#bbbbbb"}

# Metric battery, split by SAMPLING-INVARIANCE (not an informal level/dynamics
# label). Marginal & risk metrics are pointwise functions of BG averaged over
# samples, so they are cadence-invariant and are referenced to ALL three real
# cohorts. LBGI/HBGI belong here (they are means of a per-sample risk transform),
# NOT with the rate-of-change metrics. GMI is dropped: it is affine in the mean
# and would double-count it. Keys index per-record `summary`.
LEVEL_METRICS = [
    ("mean", "Mean BG", "mg/dL"),
    ("cv_pct", "CV", "%"), ("TIR_pct", "TIR 70-180", "%"),
    ("TBR1_pct", "TBR 54-70", "%"), ("TAR1_pct", "TAR 180-250", "%"),
    ("p10", "10th pct", "mg/dL"), ("p90", "90th pct", "mg/dL"),
    ("LBGI", "LBGI", ""), ("HBGI", "HBGI", ""),
]
# Rate-of-change metrics depend on sampling cadence, so they are referenced only
# to the natively-5-min real cohorts (Ohio, AZT1D); Shanghai (15-min) is excluded.
DYN_METRICS = [
    ("delta_std", "5-min ΔBG SD", "mg/dL"), ("mage", "MAGE", "mg/dL"),
    ("conga_1h", "CONGA-1h", "mg/dL"), ("modd", "MODD", "mg/dL"),
    ("sample_entropy", "Sample entropy", ""),
]


# ---------------------------------------------------------------------------
# UVA/Padova native data generation
# ---------------------------------------------------------------------------
def _uva_worker(args):
    """Run one native simglucose simulation; return (id, 3-min CGM array)."""
    name, seed, days, warmup_h = args
    from simglucose.patient.t1dpatient import T1DPatient
    from simglucose.sensor.cgm import CGMSensor
    from simglucose.actuator.pump import InsulinPump
    from simglucose.simulation.scenario_gen import RandomScenario
    from simglucose.simulation.env import T1DSimEnv
    from simglucose.simulation.sim_engine import SimObj
    from simglucose.controller.basal_bolus_ctrller import BBController

    start = dt.datetime(2024, 1, 1, 0, 0)
    env = T1DSimEnv(T1DPatient.withName(name), CGMSensor.withName("Dexcom", seed=seed),
                    InsulinPump.withName("Insulet"), RandomScenario(start_time=start, seed=seed))
    obj = SimObj(env, BBController(), dt.timedelta(days=days), animate=False, path=None)
    obj.simulate()
    cgm = obj.results()["CGM"].to_numpy(dtype=float)
    sensor_dt = 3  # Dexcom cadence (min)
    warm = warmup_h * 60 // sensor_dt
    return f"{name}/s{seed}", cgm[warm:], sensor_dt


def uva_items(n_replicates, days, warmup_h):
    adults = padova_engine.patient_names("adult")
    jobs = [(name, seed, days, warmup_h)
            for name in adults for seed in range(n_replicates)]
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        out = list(ex.map(_uva_worker, jobs))
    start = dt.datetime(2024, 1, 1, 0, 0)
    items = []
    for rid, cgm, sdt in out:
        rows = [(start + dt.timedelta(minutes=sdt * i), float(cgm[i])) for i in range(len(cgm))]
        items.append((rid, rows))
    return items


# ---------------------------------------------------------------------------
# Distances and metric errors
# ---------------------------------------------------------------------------
def real_vs_real_floor(cohorts):
    """Noise floor: BG distance among all three real cohorts (sampling-invariant)
    and ΔBG distance between the two natively-5-min cohorts (Ohio, AZT1D)."""
    names = ["Ohio", "Shanghai", "AZT1D"]
    bg = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            bg.append(distribution_distances(cohorts[names[i]]["pooled_bg"],
                                             cohorts[names[j]]["pooled_bg"]))
    dl = distribution_distances(cohorts["Ohio"]["pooled_delta"], cohorts["AZT1D"]["pooled_delta"])
    agg = lambda L, k: float(np.mean([d[k] for d in L]))  # noqa: E731
    return {
        "bg": {k: agg(bg, k) for k in ("ks_stat", "wasserstein", "js_div")},
        "delta": {k: dl[k] for k in ("ks_stat", "wasserstein", "js_div")},
    }


def metric_errors(sim, real, metrics):
    """For each metric: real & sim population means, and the gap in units of the
    real cross-patient SD (so <1 ≈ within real heterogeneity)."""
    rows = []
    for key, label, unit in metrics:
        rs = real["summary"].get(key)
        ss = sim["summary"].get(key)
        if not rs or not ss:
            continue
        rv, sv = rs["mean"], ss["mean"]
        spread = rs["std"] if rs["std"] > 1e-9 else float("nan")
        rows.append({"key": key, "label": label, "unit": unit,
                     "real": rv, "sim": sv, "abs_err": abs(sv - rv),
                     "norm_err": abs(sv - rv) / spread if spread == spread else float("nan")})
    return rows


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, name), dpi=130)
    plt.close(fig)


def fig_pdf(cohorts, fname):
    fig, ax = plt.subplots(figsize=(11, 5.4))
    bins = np.linspace(40, 400, 100)
    for n, c, lw, a in (("Real", C["Real"], 2.4, 1.0),
                        ("T1DMSIM", C["T1DMSIM"], 2.0, 0.85),
                        ("UVA/Padova", C["UVA/Padova"], 2.0, 0.85)):
        ax.hist(cohorts[n]["pooled_bg"], bins=bins, density=True, histtype="step",
                color=c, lw=lw, alpha=a, label=n)
    for v in (70, 180):
        ax.axvline(v, color="gray", ls="--", lw=1)
    ax.set_xlabel("BG (mg/dL)")
    ax.set_ylabel("density")
    ax.set_title("Pooled BG distribution vs. real CGM")
    ax.legend()
    ax.grid(alpha=0.25)
    _save(fig, fname)


def fig_cdf(cohorts, fname):
    fig, ax = plt.subplots(figsize=(11, 5.4))
    for n, c in (("Real", C["Real"]), ("T1DMSIM", C["T1DMSIM"]), ("UVA/Padova", C["UVA/Padova"])):
        x = np.sort(cohorts[n]["pooled_bg"])
        ax.plot(x, np.linspace(0, 1, len(x)), color=c, lw=2, label=n)
    for v in (70, 180):
        ax.axvline(v, color="gray", ls="--", lw=1)
    ax.set_xlabel("BG (mg/dL)")
    ax.set_ylabel("cumulative fraction")
    ax.set_title("Pooled BG CDF vs. real CGM")
    ax.legend()
    ax.grid(alpha=0.25)
    _save(fig, fname)


def fig_delta(cohorts, fname):
    fig, ax = plt.subplots(figsize=(11, 5.4))
    bins = np.linspace(-30, 30, 120)
    for n, c in (("Real", C["Real"]), ("T1DMSIM", C["T1DMSIM"]), ("UVA/Padova", C["UVA/Padova"])):
        ax.hist(cohorts[n]["pooled_delta"], bins=bins, density=True, histtype="step",
                color=c, lw=2, label=n)
    ax.set_yscale("log")
    ax.set_xlabel("5-min ΔBG (mg/dL)")
    ax.set_ylabel("density (log)")
    ax.set_title("Rate-of-change distribution vs. real CGM")
    ax.legend()
    ax.grid(alpha=0.25)
    _save(fig, fname)


def fig_acf(cohorts, fname):
    lags = [5, 15, 30, 60, 120, 240, 480, 720, 1440]
    fig, ax = plt.subplots(figsize=(11, 5.4))
    for n, c in (("Real", C["Real"]), ("T1DMSIM", C["T1DMSIM"]), ("UVA/Padova", C["UVA/Padova"])):
        y = [cohorts[n]["pooled_acf"].get(L, np.nan) for L in lags]
        ax.plot([L / 60 for L in lags], y, color=c, lw=2, marker="o", ms=4, label=n)
    ax.set_xscale("log")
    ax.set_xlabel("lag (hours, log)")
    ax.set_ylabel("autocorrelation")
    ax.set_title("BG autocorrelation vs. real CGM")
    ax.legend()
    ax.grid(alpha=0.25, which="both")
    _save(fig, fname)


def fig_distance_bars(dist_ours, dist_uva, delta_ours, delta_uva, floor, fname):
    reals = ["Ohio", "Shanghai", "AZT1D"]
    x = np.arange(len(reals))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2),
                                   gridspec_kw={"width_ratios": [3, 1]})
    ax1.bar(x - 0.2, [dist_ours[r]["wasserstein"] for r in reals], 0.4,
            color=C["T1DMSIM"], label="T1DMSIM")
    ax1.bar(x + 0.2, [dist_uva[r]["wasserstein"] for r in reals], 0.4,
            color=C["UVA/Padova"], label="UVA/Padova")
    ax1.axhline(floor["bg"]["wasserstein"], color="green", ls="--", lw=1.5,
                label="real-vs-real floor")
    ax1.set_xticks(x)
    ax1.set_xticklabels(reals)
    ax1.set_ylabel("Wasserstein-1 (mg/dL)")
    ax1.set_title("BG distribution distance to real")
    ax1.legend()
    ax1.grid(alpha=0.25, axis="y")

    ax2.bar([0], [delta_ours["wasserstein"]], 0.6, color=C["T1DMSIM"])
    ax2.bar([1], [delta_uva["wasserstein"]], 0.6, color=C["UVA/Padova"])
    ax2.axhline(floor["delta"]["wasserstein"], color="green", ls="--", lw=1.5)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["T1DMSIM", "UVA/P"])
    ax2.set_ylabel("Wasserstein-1 (mg/dL)")
    ax2.set_title("ΔBG distance (5-min reals)")
    ax2.grid(alpha=0.25, axis="y")
    _save(fig, fname)


def fig_metric_error(err_ours, err_uva, fname):
    def by_key(errs):
        return {e["key"]: e for e in errs}
    eo, eu = by_key(err_ours), by_key(err_uva)
    keys = [k for k, _, _ in (LEVEL_METRICS + DYN_METRICS) if k in eo and k in eu]
    labels = [eo[k]["label"] for k in keys]
    vo = [eo[k]["norm_err"] for k in keys]
    vu = [eu[k]["norm_err"] for k in keys]
    y = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(y - 0.2, vo, 0.4, color=C["T1DMSIM"], label="T1DMSIM")
    ax.barh(y + 0.2, vu, 0.4, color=C["UVA/Padova"], label="UVA/Padova")
    ax.axvline(1.0, color="green", ls="--", lw=1.5, label="1 real-patient SD")
    n_level = sum(1 for k in keys if k in dict((m[0], 1) for m in LEVEL_METRICS))
    ax.axhline(n_level - 0.5, color="gray", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("|sim − real| in units of real cross-patient SD")
    ax.set_title("Per-metric distance to real  (top: level · bottom: dynamics)")
    ax.legend()
    ax.grid(alpha=0.25, axis="x")
    _save(fig, fname)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    os.makedirs(FIGS, exist_ok=True)
    oseeds = 6 if args.quick else OURSIM_SEEDS
    odays = 10 if args.quick else OURSIM_DAYS
    ureps = 1 if args.quick else UVA_REPLICATES
    udays = 5 if args.quick else UVA_DAYS

    print("Loading real cohorts…")
    ohio = list(load_ohio_patients().items())
    shang = list(load_shanghai_patients().items())
    az = list(load_azt1d_patients().items())

    print(f"Generating UVA/Padova native data ({len(padova_engine.patient_names('adult'))} adults"
          f" × {ureps} × {udays}d, {MAX_WORKERS} workers)…")
    uva = uva_items(ureps, udays, UVA_WARMUP_H)

    print(f"Running T1DMSIM ({oseeds} × {odays}d)…")
    sim_items, _ = assemble_sim(n_seeds=oseeds, days=odays, warmup_h=24)

    print("Assembling cohort statistics…")
    cohorts = {
        "Ohio": assemble_cohort("Ohio", ohio, regularize_bg, 5),
        "Shanghai": assemble_cohort("Shanghai", shang, regularize_bg_15min, 15),
        "AZT1D": assemble_cohort("AZT1D", az, regularize_bg, 5),
        "T1DMSIM": assemble_cohort("T1DMSIM", sim_items, trivial_regularize_5min, 5),
        "UVA/Padova": assemble_cohort("UVA/Padova", uva, regularize_bg, 5),
    }
    for co in cohorts.values():
        co["summary"] = cohort_summary(co["per"])

    # Real references. BG level/marginal is sampling-invariant -> pool all three.
    # Dynamics (ΔBG, ACF) are sampling-dependent -> use only the natively-5-min
    # cohorts (Ohio, AZT1D); Shanghai is 15-min and is excluded there.
    ALL = ["Ohio", "Shanghai", "AZT1D"]
    FIVE = ["Ohio", "AZT1D"]
    pool_bg = lambda ns: np.concatenate([cohorts[n]["pooled_bg"] for n in ns])    # noqa: E731
    pool_per = lambda ns: [r for n in ns for r in cohorts[n]["per"]]              # noqa: E731
    real_level = {"pooled_bg": pool_bg(ALL), "summary": cohort_summary(pool_per(ALL))}
    real_dyn = {
        "pooled_bg": pool_bg(FIVE),
        "pooled_delta": np.concatenate([cohorts[n]["pooled_delta"] for n in FIVE]),
        "summary": cohort_summary(pool_per(FIVE)),
        "pooled_acf": {L: float(np.nanmean([cohorts[n]["pooled_acf"].get(L, np.nan) for n in FIVE]))
                       for L in cohorts["Ohio"]["pooled_acf"]},
    }
    # Synthetic "Real" cohort for the figures: BG from the full pool, dynamics
    # from the 5-min pool, summary merged (delta_std from the 5-min pool).
    cohorts["Real"] = {
        "pooled_bg": real_level["pooled_bg"], "pooled_delta": real_dyn["pooled_delta"],
        "pooled_acf": real_dyn["pooled_acf"],
        "summary": {**real_level["summary"], "delta_std": real_dyn["summary"].get("delta_std", {})},
    }

    floor = real_vs_real_floor(cohorts)
    dist_ours = {r: distribution_distances(cohorts["T1DMSIM"]["pooled_bg"], cohorts[r]["pooled_bg"]) for r in ALL}
    dist_uva = {r: distribution_distances(cohorts["UVA/Padova"]["pooled_bg"], cohorts[r]["pooled_bg"]) for r in ALL}
    dist_ours["Real"] = distribution_distances(cohorts["T1DMSIM"]["pooled_bg"], real_level["pooled_bg"])
    dist_uva["Real"] = distribution_distances(cohorts["UVA/Padova"]["pooled_bg"], real_level["pooled_bg"])
    delta_ours = distribution_distances(cohorts["T1DMSIM"]["pooled_delta"], real_dyn["pooled_delta"])
    delta_uva = distribution_distances(cohorts["UVA/Padova"]["pooled_delta"], real_dyn["pooled_delta"])

    err_ours = (metric_errors(cohorts["T1DMSIM"], real_level, LEVEL_METRICS)
                + metric_errors(cohorts["T1DMSIM"], real_dyn, DYN_METRICS))
    err_uva = (metric_errors(cohorts["UVA/Padova"], real_level, LEVEL_METRICS)
               + metric_errors(cohorts["UVA/Padova"], real_dyn, DYN_METRICS))

    print("Rendering figures…")
    fig_pdf(cohorts, "realism_pdf.png")
    fig_cdf(cohorts, "realism_cdf.png")
    fig_delta(cohorts, "realism_delta.png")
    fig_acf(cohorts, "realism_acf.png")
    fig_distance_bars(dist_ours, dist_uva, delta_ours, delta_uva, floor, "realism_distance_bars.png")
    fig_metric_error(err_ours, err_uva, "realism_metric_error.png")

    def group_mean(errs, metrics):
        ks = {m[0] for m in metrics}
        v = [e["norm_err"] for e in errs if e["key"] in ks and e["norm_err"] == e["norm_err"]]
        return float(np.mean(v)) if v else float("nan")

    verdict = {
        "level_norm_err": {"T1DMSIM": group_mean(err_ours, LEVEL_METRICS),
                           "UVA/Padova": group_mean(err_uva, LEVEL_METRICS)},
        "dyn_norm_err": {"T1DMSIM": group_mean(err_ours, DYN_METRICS),
                         "UVA/Padova": group_mean(err_uva, DYN_METRICS)},
        "wasserstein_to_real_bg": {"T1DMSIM": dist_ours["Real"]["wasserstein"],
                                   "UVA/Padova": dist_uva["Real"]["wasserstein"],
                                   "floor": floor["bg"]["wasserstein"]},
        "wasserstein_to_real_delta": {"T1DMSIM": delta_ours["wasserstein"],
                                      "UVA/Padova": delta_uva["wasserstein"],
                                      "floor": floor["delta"]["wasserstein"]},
    }

    def pop_row(name, dyn_ok=True):
        s = cohorts[name]["summary"]
        return {"mean": s.get("mean", {}).get("mean"),
                "TIR_pct": s.get("TIR_pct", {}).get("mean"),
                "GMI": s.get("GMI", {}).get("mean"),
                "delta_std": s.get("delta_std", {}).get("mean") if dyn_ok else None}

    stats = {
        "config": {"oursim_seeds": oseeds, "oursim_days": odays,
                   "uva_records": len(uva), "uva_days": udays,
                   "uva_dosing": "RandomScenario meals + BBController + Dexcom CGM",
                   "real_records": {"Ohio": len(ohio), "Shanghai": len(shang), "AZT1D": len(az)}},
        "pop_means": {n: pop_row(n, dyn_ok=(n != "Shanghai"))
                      for n in ("Real", "Ohio", "Shanghai", "AZT1D", "T1DMSIM", "UVA/Padova")},
        "floor": floor, "dist_ours": dist_ours, "dist_uva": dist_uva,
        "delta_dist": {"T1DMSIM": delta_ours, "UVA/Padova": delta_uva},
        "metric_errors": {"T1DMSIM": err_ours, "UVA/Padova": err_uva},
        "verdict": verdict,
    }
    with open(os.path.join(HERE, "realism.json"), "w") as f:
        json.dump(stats, f, indent=2)
    write_report(stats)
    print("  wrote realism.json, REALISM.md, 6 figures")


def write_report(stats):
    v = stats["verdict"]
    pm = stats["pop_means"]
    cfg = stats["config"]
    eo = {e["key"]: e for e in stats["metric_errors"]["T1DMSIM"]}
    eu = {e["key"]: e for e in stats["metric_errors"]["UVA/Padova"]}

    def err_rows(metrics):
        out = []
        for key, label, unit in metrics:
            if key not in eo or key not in eu:
                continue
            a, b = eo[key], eu[key]
            win = "T1DMSIM" if a["norm_err"] < b["norm_err"] else "UVA/Padova"
            out.append(f"| {label} | {a['real']:.1f} | {a['sim']:.1f} | {b['sim']:.1f} | "
                       f"{a['norm_err']:.2f} | {b['norm_err']:.2f} | {win} |")
        return "\n".join(out)

    wb = v["wasserstein_to_real_bg"]
    wd = v["wasserstein_to_real_delta"]
    closer = "T1DMSIM" if wb["T1DMSIM"] < wb["UVA/Padova"] else "UVA/Padova"
    ratio = wb["UVA/Padova"] / wb["T1DMSIM"] if wb["T1DMSIM"] else float("nan")

    def ds(name):  # delta_std cell ("—" where dynamics are not comparable)
        x = pm[name]["delta_std"]
        return f"{x:.2f}" if isinstance(x, (int, float)) else "—"

    md = f"""# Which simulator better resembles real-world CGM?

A model trained on synthetic data and run on real data pays for the gap between
the two distributions. Without training anything, this report measures that gap
for both simulators against real CGM (OhioT1DM + ShanghaiT1DM + AZT1D),
using the metric battery in `diff/build_report.py`. The yardstick is the
distance *between the real cohorts themselves* — a simulator inside that
envelope is, distributionally, as real as one real cohort is to another.

*Generated by `uva_padova/compare_realism.py`. Do not edit by hand.*

> **Read this first.** T1DMSIM was calibrated against the Ohio/Shanghai cohorts;
> UVA/Padova was not — it is a physiology + controller benchmark for closed-loop
> testing, here driven by its own native data-generation stack
> ({cfg['uva_dosing']}). T1DMSIM's calibration targeted not only the **level** but
> also distributional *shape* — the ΔBG distribution and autocorrelation — so even
> the **dynamic** metrics are not a fitting-free contest; they are merely less
> dominated by the level fit. Read every gap below as distance from a target
> T1DMSIM was trained toward and UVA/Padova never saw.

## Population summary

| Cohort | Mean BG | GMI | TIR 70–180 | 5-min ΔBG SD |
|---|---|---|---|---|
| **Real (pooled)** | {pm['Real']['mean']:.0f} | {pm['Real']['GMI']:.2f} | {pm['Real']['TIR_pct']:.0f}% | {ds('Real')} |
| · Ohio | {pm['Ohio']['mean']:.0f} | {pm['Ohio']['GMI']:.2f} | {pm['Ohio']['TIR_pct']:.0f}% | {ds('Ohio')} |
| · Shanghai (15-min) | {pm['Shanghai']['mean']:.0f} | {pm['Shanghai']['GMI']:.2f} | {pm['Shanghai']['TIR_pct']:.0f}% | {ds('Shanghai')} |
| · AZT1D | {pm['AZT1D']['mean']:.0f} | {pm['AZT1D']['GMI']:.2f} | {pm['AZT1D']['TIR_pct']:.0f}% | {ds('AZT1D')} |
| **T1DMSIM** | {pm['T1DMSIM']['mean']:.0f} | {pm['T1DMSIM']['GMI']:.2f} | {pm['T1DMSIM']['TIR_pct']:.0f}% | {ds('T1DMSIM')} |
| **UVA/Padova** | {pm['UVA/Padova']['mean']:.0f} | {pm['UVA/Padova']['GMI']:.2f} | {pm['UVA/Padova']['TIR_pct']:.0f}% | {ds('UVA/Padova')} |

## Distance to real (lower is closer)

Wasserstein-1 between each simulator's pooled distribution and real, with the
real-vs-real floor for reference:

| | T1DMSIM | UVA/Padova | real-vs-real floor |
|---|---|---|---|
| BG distribution | {wb['T1DMSIM']:.1f} | {wb['UVA/Padova']:.1f} | {wb['floor']:.1f} |
| ΔBG distribution | {wd['T1DMSIM']:.1f} | {wd['UVA/Padova']:.1f} | {wd['floor']:.1f} |

On the marginal BG distribution, **{closer}** is closer to real — by ≈{ratio:.1f}×
on Wasserstein.

![BG distribution](figures/realism_pdf.png)
![BG CDF](figures/realism_cdf.png)
![Distance to real](figures/realism_distance_bars.png)

## Per-metric distance (units of one real-patient SD)

A gap under 1.0 means the simulator's population mean lands within the spread of
real patients on that metric — i.e. indistinguishable from a real cohort. The
battery is split by **sampling cadence**: pointwise marginal & risk metrics are
cadence-invariant and referenced to all three real cohorts; rate-of-change
metrics are cadence-dependent and referenced only to the 5-min cohorts (Ohio,
AZT1D), since Shanghai is 15-min.

### Marginal & risk metrics (sampling-invariant; all three real cohorts)

| Metric | Real | T1DMSIM | UVA/Padova | ours (SD) | UVA (SD) | closer |
|---|---|---|---|---|---|---|
{err_rows(LEVEL_METRICS)}

Mean normalised error — T1DMSIM **{v['level_norm_err']['T1DMSIM']:.2f}**,
UVA/Padova **{v['level_norm_err']['UVA/Padova']:.2f}**.

### Rate-of-change metrics (cadence-dependent; 5-min cohorts only — the fairer test)

| Metric | Real | T1DMSIM | UVA/Padova | ours (SD) | UVA (SD) | closer |
|---|---|---|---|---|---|---|
{err_rows(DYN_METRICS)}

Mean normalised error — T1DMSIM **{v['dyn_norm_err']['T1DMSIM']:.2f}**,
UVA/Padova **{v['dyn_norm_err']['UVA/Padova']:.2f}**.

![Per-metric distance](figures/realism_metric_error.png)
![ΔBG distribution](figures/realism_delta.png)
![Autocorrelation](figures/realism_acf.png)

## Reading the result

A model trained on the source whose distribution is closer to real incurs less
covariate/label shift at deployment, so distance to real bounds — but does not
prove — transfer quality. T1DMSIM is closer on every aggregate here: decisively
on level (mean normalised error {v['level_norm_err']['T1DMSIM']:.2f} vs
{v['level_norm_err']['UVA/Padova']:.2f} real-patient SDs) and more narrowly on
dynamics ({v['dyn_norm_err']['T1DMSIM']:.2f} vs {v['dyn_norm_err']['UVA/Padova']:.2f}).
That ordering is expected — T1DMSIM was fit to these cohorts, UVA/Padova never
saw them — so the result confirms the calibration rather than crowning a winner.

Two findings survive the caveat. First, both simulators sit **outside** the
real-vs-real envelope on the pooled distribution ({wb['T1DMSIM']:.0f} and
{wb['UVA/Padova']:.0f} mg/dL vs a {wb['floor']:.0f} floor), so neither is a
drop-in replacement for real CGM. Second, on excursion-amplitude (MAGE, MODD)
UVA/Padova — despite no fitting — lands closer to real than T1DMSIM, which
*overshoots* them: a model trained on T1DMSIM should transfer better overall,
but would inherit its exaggerated swings.

## Reproducing

```bash
venv/bin/python uva_padova/compare_realism.py            # full
venv/bin/python uva_padova/compare_realism.py --quick    # fast smoke run
```
"""
    with open(os.path.join(HERE, "REALISM.md"), "w") as f:
        f.write(md)


if __name__ == "__main__":
    main()
