"""Generate the inline SVG visuals embedded in the project page (index.html).

Runs the simulator, renders a wide hero trace (glucose plus a carbohydrate /
insulin event strip) and a small-multiple wall of independent seeds, then
splices both into index.html in place. Idempotent: re-running replaces the
previously injected markup.

    python scripts/make_site_figures.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from simulator import T1DMSimulator  # noqa: E402

HERO_SEED = 24
HERO_HOURS = 60
WALL_SEEDS = [8, 41, 88, 23, 17, 60, 3, 52, 34, 29, 73, 11]
WALL_HOURS = 36
WARMUP_HOURS = 48

W = 1240
TRACE_H = 200
STRIP_TOP = 224
STRIP_H = 64
H = STRIP_TOP + STRIP_H
BG_LO, BG_HI = 40.0, 330.0
HYPO, HYPER = 70.0, 180.0


def run(seed: int, hours: float, warmup: float = WARMUP_HOURS) -> dict:
    sim = T1DMSimulator(seed=seed, initial_bg=120)
    sim.generate_hours(warmup)
    return sim.generate_hours(hours)


def y_of(bg: np.ndarray | float, height: int = TRACE_H, top_pad: int = 8) -> np.ndarray:
    t = (np.clip(bg, BG_LO, BG_HI) - BG_LO) / (BG_HI - BG_LO)
    return height - t * (height - top_pad)


def path_of(xs, ys) -> str:
    return "M" + " L".join(f"{x:.0f},{y:.0f}" for x, y in zip(xs, ys))


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous True spans as (start, stop) index pairs, padded by one step
    on each side so a coloured excursion joins the base trace seamlessly."""
    out, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            out.append((max(start - 1, 0), i + 1))
            start = None
    if start is not None:
        out.append((max(start - 1, 0), len(mask)))
    return [(a, b) for a, b in out if b - a > 1]


def hero() -> tuple[str, np.ndarray]:
    d = run(HERO_SEED, HERO_HOURS)
    bg = d["bg_observed"]
    n = len(bg)
    xs = np.linspace(0, W, n)
    ys = y_of(bg)
    band_top, band_bot = float(y_of(HYPER)), float(y_of(HYPO))

    p = [
        f'<svg class="hero-svg" viewBox="0 0 {W} {H}" preserveAspectRatio="none" role="img" '
        f'aria-label="{HERO_HOURS} hours of simulated CGM for one seed, with carbohydrate '
        f'and insulin events beneath">',
        '<defs><linearGradient id="fadeR" x1="0" x2="1">'
        '<stop offset="0.88" stop-color="#fff" stop-opacity="1"/>'
        '<stop offset="1" stop-color="#fff" stop-opacity="0"/></linearGradient>'
        '<mask id="fade"><rect width="100%" height="100%" fill="url(#fadeR)"/></mask></defs>',
        '<g mask="url(#fade)">',
        f'<rect class="band" x="0" y="{band_top:.0f}" width="{W}" height="{band_bot - band_top:.0f}"/>',
        f'<line class="rule" x1="0" y1="{band_top:.0f}" x2="{W}" y2="{band_top:.0f}"/>',
        f'<line class="rule" x1="0" y1="{band_bot:.0f}" x2="{W}" y2="{band_bot:.0f}"/>',
    ]

    hod = d["hour_of_day"]
    for i in range(1, n):
        if hod[i] < hod[i - 1]:
            p.append(f'<line class="midnight" x1="{xs[i]:.0f}" y1="0" x2="{xs[i]:.0f}" y2="{TRACE_H}"/>')

    p.append(f'<path class="trace" d="{path_of(xs, ys)}"/>')
    for mask, cls in ((bg > HYPER, "hi"), (bg < HYPO, "lo")):
        for a, b in _runs(mask):
            p.append(f'<path class="seg {cls}" d="{path_of(xs[a:b], ys[a:b])}"/>')

    # event strip: carbohydrate as upward impulses, insulin as a mirrored area
    binsz = 3
    nb = n // binsz
    carb = d["total_carb"][: nb * binsz].reshape(nb, binsz).sum(1)
    ins = d["total_insulin"][: nb * binsz].reshape(nb, binsz).sum(1)
    bxs = np.linspace(0, W, nb, endpoint=False)
    mid = STRIP_TOP + STRIP_H / 2
    half = STRIP_H / 2 - 3
    p.append(f'<line class="rule" x1="0" y1="{mid:.0f}" x2="{W}" y2="{mid:.0f}"/>')
    iy = mid + ins / max(ins.max(), 1e-6) * half
    p.append(f'<path class="ins-area" d="{path_of(bxs, iy)} L{W},{mid:.0f} L0,{mid:.0f} Z"/>')
    bw = max(W / nb * 0.55, 1.6)
    for i in range(nb):
        if carb[i] > 0.05:
            h = carb[i] / max(carb.max(), 1e-6) * half
            p.append(f'<rect class="carb" x="{bxs[i]:.0f}" y="{mid - h:.0f}" '
                     f'width="{bw:.1f}" height="{h:.0f}"/>')
    p.append("</g></svg>")
    return "\n".join(p), bg


def wall(cw: int = 280, ch: int = 64, step: int = 2) -> str:
    cells = []
    for seed in WALL_SEEDS:
        bg = run(seed, WALL_HOURS)["bg_observed"]
        sub = bg[::step]
        xs = np.linspace(0, cw, len(sub))
        ys = y_of(sub, height=ch - 3, top_pad=6)
        bt, bb = float(y_of(HYPER, ch - 3, 6)), float(y_of(HYPO, ch - 3, 6))
        tir = float(((bg >= HYPO) & (bg <= HYPER)).mean() * 100)
        segs = "".join(
            f'<path class="seg {cls}" d="{path_of(xs[a:b], ys[a:b])}"/>'
            for mask, cls in ((sub > HYPER, "hi"), (sub < HYPO, "lo"))
            for a, b in _runs(mask)
        )
        cells.append(
            '<div class="cell"><div class="hd">'
            f'<span class="sd">seed {seed}</span>'
            f'<span class="tir">{bg.mean():.0f} mg/dL · {tir:.0f}% TIR</span></div>'
            f'<svg class="mini" viewBox="0 0 {cw} {ch}" preserveAspectRatio="none" role="img" '
            f'aria-label="seed {seed}: mean {bg.mean():.0f} milligrams per decilitre, '
            f'{tir:.0f} percent in range">'
            f'<rect class="band" x="0" y="{bt:.0f}" width="{cw}" height="{bb - bt:.0f}"/>'
            f'<path class="trace" d="{path_of(xs, ys)}"/>{segs}</svg></div>'
        )
    return '<div class="wall">\n' + "\n".join(cells) + "\n</div>"


def inject(page: Path, hero_svg: str, wall_html: str) -> None:
    src = page.read_text(encoding="utf-8")
    for what, patterns, replacement in (
        ("hero", (r'<svg class="hero-svg".*?</svg>', r"<!--HERO-->"), hero_svg),
        ("wall", (r'<div class="wall">.*?\n</div>', r"<!--WALL-->"), wall_html),
    ):
        for pattern in patterns:
            if re.search(pattern, src, flags=re.S):
                src = re.sub(pattern, lambda _m, r=replacement: r, src, count=1, flags=re.S)
                print(f"  injected {what}")
                break
        else:
            raise SystemExit(f"no injection site for {what} in {page}")
    page.write_text(src, encoding="utf-8")


if __name__ == "__main__":
    page = Path(__file__).resolve().parent.parent / "index.html"
    hero_svg, bg = hero()
    wall_html = wall()
    inject(page, hero_svg, wall_html)
    tir = float(((bg >= HYPO) & (bg <= HYPER)).mean() * 100)
    print(f"hero seed {HERO_SEED}: mean {bg.mean():.0f} mg/dL, {tir:.0f}% TIR, "
          f"min {bg.min():.0f}, max {bg.max():.0f}")
    print(f"{page.name}: {page.stat().st_size / 1024:.1f} KB")
