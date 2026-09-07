"""Hypo-onset profile of a simulator cache or a CGM CSV.

usage: onset_profile.py CACHE_DIR | --csv FILE [--bg-col bg_mgdl] [--time-col time_local]
Per onset (first step < 70): BG two hours earlier, hour, duration below 70, nadir.
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

THRESHOLD = 70.0
SEVERE = 55.0
LEAD_STEPS = 24          # two hours at 5 min
HIGH_ORIGIN = 130.0
BRIEF_MIN = 15


def load_cache(path: Path) -> list[tuple[np.ndarray, np.ndarray]]:
    rows = []
    for name in ("bg_observed", "hour_of_day"):
        npy = path / f"{name}.npy"
        if npy.exists():
            rows.append(np.load(npy))
            continue
        import blosc2
        rows.append(np.asarray(blosc2.open(str(path / f"{name}.b2nd"), mode="r")[:]))
    bg, hour = rows
    return [(bg[i].astype(np.float64), hour[i].astype(np.float64)) for i in range(bg.shape[0])]


def load_csv(path: Path, bg_col: str, time_col: str) -> list[tuple[np.ndarray, np.ndarray]]:
    with open(path) as f:
        recs = list(csv.DictReader(f))
    bg = np.array([float(r[bg_col]) if r[bg_col] else np.nan for r in recs])
    hour = np.array([int(r[time_col][11:13]) + int(r[time_col][14:16]) / 60.0 for r in recs])
    return [(bg, hour)]


def onsets(bg: np.ndarray, hour: np.ndarray) -> list[tuple[float, float, int, float]]:
    out = []
    low = bg < THRESHOLD
    for o in np.where(low[1:] & ~low[:-1])[0] + 1:
        if o < LEAD_STEPS or not np.all(np.isfinite(bg[o - LEAD_STEPS:o + 1])):
            continue
        e = int(o)
        while e < len(bg) and low[e]:
            e += 1
        out.append((float(bg[o - LEAD_STEPS]), float(hour[o]), 5 * (e - int(o)), float(bg[o:e].min())))
    return out


def profile(rows: list[tuple[np.ndarray, np.ndarray]]) -> dict[str, float]:
    ev = np.array([x for bg, hour in rows for x in onsets(bg, hour)], dtype=np.float64)
    steps = sum(int(np.isfinite(bg).sum()) for bg, _ in rows)
    if len(ev) == 0:
        return {"onsets": 0, "steps": steps}
    origin, hour, dur, nadir = ev.T
    return {
        "onsets": int(len(ev)),
        "steps": steps,
        "per_week": len(ev) / steps * 288 * 7,
        "brief_pct": 100.0 * float(np.mean(dur <= BRIEF_MIN)),
        "duration_median_min": float(np.median(dur)),
        "severe_pct": 100.0 * float(np.mean(nadir < SEVERE)),
        "nadir_median": float(np.median(nadir)),
        "from_high_pct": 100.0 * float(np.mean(origin >= HIGH_ORIGIN)),
        "origin_median": float(np.median(origin)),
        "awake_pct": 100.0 * float(np.mean((hour >= 7) & (hour < 23))),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cache", nargs="?", help="cache directory (bg_observed + hour_of_day)")
    ap.add_argument("--csv", help="CGM CSV on the 5-min grid")
    ap.add_argument("--bg-col", default="bg_mgdl")
    ap.add_argument("--time-col", default="time_local", help="ISO local time column")
    args = ap.parse_args()
    if bool(args.cache) == bool(args.csv):
        ap.error("give a cache directory or --csv, not both")
    rows = load_csv(Path(args.csv), args.bg_col, args.time_col) if args.csv else load_cache(Path(args.cache))
    p = profile(rows)
    if p["onsets"] == 0:
        print("no onsets")
        return
    print(f"onsets {p['onsets']}  ({p['per_week']:.1f}/week over {p['steps'] / 288:.1f} days)")
    print(f"duration <= {BRIEF_MIN} min      {p['brief_pct']:5.1f} %   median {p['duration_median_min']:.0f} min")
    print(f"nadir < {SEVERE:.0f}             {p['severe_pct']:5.1f} %   median {p['nadir_median']:.0f} mg/dL")
    print(f"origin >= {HIGH_ORIGIN:.0f} (2 h before) {p['from_high_pct']:5.1f} %   median {p['origin_median']:.0f} mg/dL")
    print(f"awake (07-23 h)         {p['awake_pct']:5.1f} %")


if __name__ == "__main__":
    main()
