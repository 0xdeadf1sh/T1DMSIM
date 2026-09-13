"""Simulator-only dynamics: dose-response, bolus context, rescue carbs, carb balance, coupling.

A probe forks one simulator at a quiet point into arms differing only by an injected bolus; both
arms share the physiology noise stream. A frozen arm also stops every behaviour."""
from __future__ import annotations

import copy

import numpy as np

import simulator as S

DT = S.DT_MINUTES
DOSES_U = (1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0)
HORIZONS_MIN = (60, 120, 180, 240)
HORIZON_STEPS = 48
QUIET_STEPS = 24  # 2 h without carbs or bolus either side of a fork
ISOLATION_STEPS = 36  # isolated: under ISOLATION_CARB_G of meal carbs within 3 h either side
ISOLATION_CARB_G = 2.0
FOLLOWED_CARB_G = 20.0  # meal carbs follow when this many grams appear in the next 3 h
BANDS_MG_DL = ((90.0, 120.0), (160.0, 220.0))
DOSE_BINS_U = (0.0, 2.0, 4.0, 8.0, np.inf)
MIN_BIN_EVENTS = 20
MI_BIN_STEPS = 6
MI_MAX_LAG_BINS = 6
MI_NULL_WEEK_SHIFTS = 4


def freeze(sim: S.T1DMSimulator) -> None:
    """Stop all behaviour: scheduled events, new days, rescue carbs, corrections, basal suspend."""
    sim._pending_events = []
    sim._plan_day = lambda: None
    sim._generate_day_events = lambda: None
    sim._check_and_correct = lambda idx: None


def inject_bolus(sim: S.T1DMSimulator, units: float) -> None:
    p = sim.patient
    k, theta, dur = S.bolus_pk_for_dose(units, p.bolus_gamma_k, p.bolus_gamma_theta,
                                        p.bolus_dia_base_hours)
    sim.inject_curve(S.gamma_curve(units, k, theta, dur), sim.state.current_idx, "bolus", "probe")


def quiet_forks(n_seeds: int, probes_per_seed: int = 6, warmup_h: float = 48.0,
                span_days: float = 20.0) -> list[S.T1DMSimulator]:
    """Simulators paused at the first quiet step of each of probes_per_seed stretches."""
    forks = []
    stride = int(span_days * S.STEPS_PER_DAY / probes_per_seed)
    for seed in range(n_seeds):
        base = S.T1DMSimulator(seed=seed, initial_bg=120.0)
        for _ in range(int(warmup_h * 60 / DT)):
            base.generate()
        for _ in range(probes_per_seed):
            scout = copy.deepcopy(base)
            steps = [scout.generate() for _ in range(stride)]
            carb = np.array([s["total_carb"] for s in steps])
            bol = np.array([s["bolus_insulin"] for s in steps])
            for i in range(QUIET_STEPS, stride - HORIZON_STEPS - QUIET_STEPS):
                w = slice(i - QUIET_STEPS, i + QUIET_STEPS)
                if carb[w].sum() < 0.5 and bol[w].sum() < 0.05:
                    fork = copy.deepcopy(base)
                    for _ in range(i):
                        fork.generate()
                    forks.append(fork)
                    break
            base = scout
    return forks


def _arm(fork: S.T1DMSimulator, units: float, frozen: bool) -> tuple[np.ndarray, float]:
    """True BG over HORIZON_STEPS and rescue grams injected, for one arm of a probe."""
    sim = copy.deepcopy(fork)
    idx0 = sim.state.current_idx
    if units > 0.0:
        inject_bolus(sim, units)
    if frozen:
        freeze(sim)
    bg = np.array([sim.generate()["bg"] for _ in range(HORIZON_STEPS)])
    rescue = sum(g for i, kind, g, _ in sim.injection_log
                 if i >= idx0 and kind == "correction_carb")
    return bg, float(rescue)


def dose_response(forks: list[S.T1DMSimulator], doses: tuple = DOSES_U) -> dict:
    """Open-loop ΔBG per probe and dose; monotone share, per-unit and marginal response."""
    delta = np.empty((len(forks), len(doses), HORIZON_STEPS))
    for p, fork in enumerate(forks):
        base, _ = _arm(fork, 0.0, True)
        for j, units in enumerate(doses):
            delta[p, j] = _arm(fork, units, True)[0] - base
    u = np.asarray(doses)

    def at(minutes: int) -> np.ndarray:
        return delta[:, :, minutes // DT - 1]

    one_u = at(240)[:, list(doses).index(1.0)] if 1.0 in doses else np.full(len(forks), np.nan)
    cf = np.array([f.patient.correction_factor for f in forks])
    mono2 = np.all(np.diff(at(120), axis=1) <= 1e-9, axis=1)
    mono4 = np.all(np.diff(at(240), axis=1) <= 1e-9, axis=1)
    mean4 = at(240).mean(axis=0)
    return {
        "doses_u": list(doses),
        "n_probes": len(forks),
        "bg0": [float(f.state.bg) for f in forks],
        "monotone_pct_2h": float(100.0 * mono2.mean()),
        "monotone_pct_4h": float(100.0 * mono4.mean()),
        "monotone_pct_both": float(100.0 * (mono2 & mono4).mean()),
        "mean_delta": {str(h): at(h).mean(axis=0).tolist() for h in HORIZONS_MIN},
        "per_unit_4h": (mean4 / u).tolist(),
        "marginal_per_unit_4h": (np.diff(mean4, prepend=0.0) / np.diff(u, prepend=0.0)).tolist(),
        "correction_factor_mean": float(cf.mean()),
        "drop_1u_4h_mean": float(-one_u.mean()),
        "_delta_4h": at(240),
    }


def band_response(forks: list[S.T1DMSimulator], units: float = 4.0) -> dict:
    """Open-loop per-unit ΔBG at 4 h with every fork restarted at each band's midpoint BG.

    Each fork runs in every band, so the ratio compares starting BG rather than patients."""
    bands = {}
    for lo, hi in BANDS_MG_DL:
        mid = 0.5 * (lo + hi)
        per_unit = []
        for fork in forks:
            start = copy.deepcopy(fork)
            start.state.bg = start.state.bg_observed = start._interstitial_bg = mid
            dosed = _arm(start, units, True)[0][-1]
            per_unit.append((dosed - _arm(start, 0.0, True)[0][-1]) / units)
        bands[f"{lo:.0f}-{hi:.0f}"] = {"n": len(per_unit), "per_unit_4h": float(np.mean(per_unit))}
    low, high = (v["per_unit_4h"] for v in bands.values())
    return {"units": units, "bands": bands, "low_over_high": float(low / high)}


def closed_loop_retention(forks: list[S.T1DMSimulator], units: float = 6.0) -> dict:
    """ΔBG at 4 h with behaviour live against frozen, and the rescue grams the bolus draws."""
    opened, closed, extra = [], [], []
    for fork in forks:
        opened.append(_arm(fork, units, True)[0][-1] - _arm(fork, 0.0, True)[0][-1])
        c0, r0 = _arm(fork, 0.0, False)
        c1, r1 = _arm(fork, units, False)
        closed.append(c1[-1] - c0[-1])
        extra.append(r1 - r0)
    extra = np.asarray(extra)
    return {
        "units": units,
        "open_loop_4h": float(np.mean(opened)),
        "closed_loop_4h": float(np.mean(closed)),
        "retention_pct": float(100.0 * np.mean(closed) / np.mean(opened)),
        "extra_rescue_g_mean": float(extra.mean()),
        "extra_rescue_probe_pct": float(100.0 * np.mean(extra > 1e-9)),
    }


def _log_rel(run: dict, kind: str) -> list[tuple[int, float]]:
    """(step within the run, amount) for each logged injection of one curve type."""
    start = run["start_idx"]
    return [(i - start, g) for i, k, g, _ in run["injection_log"] if k == kind and i >= start]


def bolus_context(runs: list[dict]) -> dict:
    """Bolus injections by meal-carb context; forward CGM ΔBG by context and by dose bin.

    The isolated share uses ±3 h; the isolated ΔBG rows also need no meal carbs up to +4 h."""
    groups: dict[str, list] = {"isolated": [], "carbs_follow": [], "no_carbs_follow": []}
    iso_by_dose, n_bolus, n_isolated = [], 0, 0
    reach = max(ISOLATION_STEPS, HORIZONS_MIN[-1] // DT)
    for run in runs:
        bg = np.asarray(run["bg_observed"], float)
        meal = np.asarray(run["total_carb"], float) - np.asarray(run["rescue_carb"], float)
        cum = np.concatenate(([0.0], np.cumsum(meal)))
        n = len(bg)
        for i, units in _log_rel(run, "bolus"):
            if i < ISOLATION_STEPS or i + reach >= n:
                continue
            n_bolus += 1
            fwd = np.array([bg[i - 1 + h // DT] - bg[i - 1] for h in HORIZONS_MIN])
            around = cum[i + ISOLATION_STEPS] - cum[i - ISOLATION_STEPS]
            ahead = cum[i + ISOLATION_STEPS] - cum[i]
            if around < ISOLATION_CARB_G:
                n_isolated += 1
                if cum[i + reach] - cum[i - ISOLATION_STEPS] < ISOLATION_CARB_G:
                    groups["isolated"].append(fwd)
                    iso_by_dose.append((units, fwd))
            if ahead >= FOLLOWED_CARB_G:
                groups["carbs_follow"].append(fwd)
            elif ahead < ISOLATION_CARB_G:
                groups["no_carbs_follow"].append(fwd)
    out = {
        "horizons_min": list(HORIZONS_MIN),
        "n_bolus": n_bolus,
        "isolated_pct": 100.0 * n_isolated / n_bolus if n_bolus else float("nan"),
        "groups": {k: {"n": len(v), "mean_delta": np.mean(v, axis=0).tolist() if v else []}
                   for k, v in groups.items()},
    }
    bins = []
    for lo, hi in zip(DOSE_BINS_U[:-1], DOSE_BINS_U[1:]):
        sel = [f for u, f in iso_by_dose if lo <= u < hi]
        bins.append({"lo_u": lo, "hi_u": hi, "n": len(sel),
                     "mean_delta": np.mean(sel, axis=0).tolist() if sel else []})
    out["isolated_by_dose"] = bins
    usable = [b["mean_delta"] for b in bins if b["n"] >= MIN_BIN_EVENTS]
    out["isolated_monotone_in_dose"] = bool(
        len(usable) >= 2 and np.all(np.diff(np.asarray(usable), axis=0) <= 0.0))
    return out


def rescue_stats(runs: list[dict]) -> dict:
    """Per CGM hypo episode (<70 mg/dL for ≥15 min): rescued share, latency and grams eaten."""
    episodes = rescued = 0
    latency, grams = [], []
    days = 0.0
    for run in runs:
        bg = np.asarray(run["bg_observed"], float)
        n = len(bg)
        days += n * DT / 1440.0
        rescues = _log_rel(run, "correction_carb")
        i = 0
        while i < n:
            if bg[i] >= 70.0:
                i += 1
                continue
            j = i
            while j < n and bg[j] < 70.0:
                j += 1
            if (j - i) * DT >= 15:
                episodes += 1
                inside = [(k, g) for k, g in rescues if i - 6 <= k <= j + 6]
                if inside:
                    rescued += 1
                    latency.append((inside[0][0] - i) * DT)
                    grams.append(sum(g for _, g in inside))
            i = j

    def pct(a: list, q: float) -> float:
        return float(np.percentile(a, q)) if a else float("nan")

    return {
        "episodes_per_day": episodes / days if days else float("nan"),
        "rescued_pct": 100.0 * rescued / episodes if episodes else float("nan"),
        "latency_min_median": pct(latency, 50),
        "grams_median": pct(grams, 50),
        "grams_p90": pct(grams, 90),
        "rescue_injections_per_day": sum(len(_log_rel(r, "correction_carb")) for r in runs) / days,
    }


def carb_balance(n_seeds: int = 12, days: int = 6) -> dict:
    """Largest gap between a meal's carb-channel grams and its logged grams."""
    worst, n_meals = 0.0, 0
    for seed in range(n_seeds):
        sim = S.T1DMSimulator(seed=seed)
        for day in range(days):
            if day:
                sim.state.day_number = day
                sim._plan_day()
                sim._generate_day_events()
            meals: dict = {}
            for _, kind, data in sim._pending_events:
                if kind == "carb":
                    got = meals.get(data["meal_key"], (0.0, data["meal_grams"]))[0]
                    meals[data["meal_key"]] = (got + float(np.sum(data["curve"])),
                                               data["meal_grams"])
            sim._pending_events = []
            n_meals += len(meals)
            worst = max([worst] + [abs(a - b) for a, b in meals.values()])
    return {"n_meals": n_meals, "max_abs_error_g": worst}


def _states(x: np.ndarray) -> np.ndarray:
    """0 for no amount, else 1-3 by tertile of the non-zero amounts."""
    out = np.zeros(len(x), dtype=int)
    nz = x > 0.0
    if nz.any():
        cuts = np.percentile(x[nz], [100 / 3, 200 / 3])
        out[nz] = 1 + np.searchsorted(cuts, x[nz], side="right")
    return out


def _mi_bits(a: np.ndarray, b: np.ndarray) -> float:
    joint = np.zeros((4, 4))
    np.add.at(joint, (a, b), 1.0)
    joint /= joint.sum()
    pa, pb = joint.sum(axis=1, keepdims=True), joint.sum(axis=0, keepdims=True)
    nz = joint > 0
    return float(np.sum(joint[nz] * np.log2(joint[nz] / (pa @ pb)[nz])))


def meal_bolus_coupling(runs: list[dict]) -> dict:
    """Mutual information (bits) between meal grams and bolus units in 30-min bins, lags ±3 h.

    Positive lag: bolus after meal. Bound: largest MI over every lag with the bolus series shifted
    by 1..MI_NULL_WEEK_SHIFTS whole weeks, which keeps time-of-day and day-of-week structure."""
    meal_s, bol_s = [], []
    for run in runs:
        nb = len(run["bg_observed"]) // MI_BIN_STEPS
        meal, bol = np.zeros(nb), np.zeros(nb)
        for i, g in _log_rel(run, "carb"):
            if i // MI_BIN_STEPS < nb:
                meal[i // MI_BIN_STEPS] += g
        for i, u in _log_rel(run, "bolus"):
            if i // MI_BIN_STEPS < nb:
                bol[i // MI_BIN_STEPS] += u
        meal_s.append(meal)
        bol_s.append(bol)
    splits = np.cumsum([len(m) for m in meal_s])[:-1]
    m_runs = np.split(_states(np.concatenate(meal_s)), splits)
    b_runs = np.split(_states(np.concatenate(bol_s)), splits)
    day_bins = S.STEPS_PER_DAY // MI_BIN_STEPS

    def mi_at(lag: int, shift_weeks: int) -> float:
        xa, xb = [], []
        for m, b in zip(m_runs, b_runs):
            b = np.roll(b, shift_weeks * 7 * day_bins)
            if lag >= 0:
                xa.append(m[:len(m) - lag])
                xb.append(b[lag:])
            else:
                xa.append(m[-lag:])
                xb.append(b[:len(b) + lag])
        return _mi_bits(np.concatenate(xa), np.concatenate(xb))

    lags = list(range(-MI_MAX_LAG_BINS, MI_MAX_LAG_BINS + 1))
    observed = [mi_at(L, 0) for L in lags]
    bound = max(mi_at(L, k) for L in lags for k in range(1, MI_NULL_WEEK_SHIFTS + 1))
    return {"lags_min": [L * MI_BIN_STEPS * DT for L in lags], "mi_bits": observed,
            "null_bound_bits": bound, "below_bound": bool(max(observed) <= bound)}


def compute_dynamics(raw_runs: list[tuple[str, dict]], probe_seeds: int) -> dict:
    """Every dynamics statistic; each raw run carries injection_log and start_idx."""
    runs = [d for _, d in raw_runs]
    forks = quiet_forks(probe_seeds)
    dr = dose_response(forks)
    return {
        "dose_response": {k: v for k, v in dr.items() if not k.startswith("_")},
        "band_response": band_response(forks),
        "closed_loop": closed_loop_retention(forks),
        "bolus_context": bolus_context(runs),
        "rescue": rescue_stats(runs),
        "carb_balance": carb_balance(),
        "meal_bolus_coupling": meal_bolus_coupling(runs),
    }
