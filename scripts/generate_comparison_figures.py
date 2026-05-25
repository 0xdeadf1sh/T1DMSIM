"""Regenerate the dataset-comparison figures embedded in README.md.

Outputs into ./assets/ at the repo root:
  - clinical_ranges.png   — stacked bars of TBR2/TBR1/TIR/TAR1/TAR2 per dataset
  - diurnal_bg.png        — 3-dataset overlay of mean BG by hour-of-day
  - bg_histogram.png      — overlapping pooled CGM-value histograms
  - episode_durations.png — hypo/hyper episode duration boxplots
  - postmeal_envelope.png — post-meal Δ-from-baseline median + IQR
  - 24h_traces.png        — 3-row grid of random 24h CGM windows
  - quantitative_shape.png — Δ-BG hist, ACF, sample entropy
  - nocturnal_traces.png  — 3-row grid of nocturnal windows

Run from repo root with the analysis venv:

    /tmp/uvavenv/bin/python scripts/generate_comparison_figures.py

Requires pandas / openpyxl / xlrd / matplotlib in the venv. Reads OhioT1DM and
ShanghaiT1DM raw data from the project root (gitignored, non-redistributable).
"""
import os
import shutil
import sys
from datetime import datetime, timedelta

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ASSETS = os.path.join(REPO_ROOT, 'assets')
os.makedirs(ASSETS, exist_ok=True)

sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(__file__))

from compare_all_datasets import (  # noqa: E402
    load_ohio_patients, load_shanghai_patients,
    regularize_bg, regularize_bg_15min,
    ohio_event_rates, shanghai_event_rates,
    run_pipeline, TappedSimulator,
    basic_stats, clinical_ranges, variability_metrics, episodes,
    diurnal_means, aggregate,
)

# Colour palette consistent across all panels
COL_OHIO  = '#1f77b4'  # blue
COL_SHANG = '#ff7f0e'  # orange
COL_SIM   = '#d62728'  # red


def run_3way(n_seeds=30, days=70):
    """Run the full 3-way pipeline. Returns dict with per-dataset stats."""
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

    print(f'Running T1DMSIM: {n_seeds} seeds × {days} days...')
    sim_per = []
    sim_pooled_bg = []
    sim_diurnals = []
    sim_hypo_durs = []
    sim_hyper_durs = []
    for seed in range(n_seeds):
        s = TappedSimulator(seed=seed, initial_bg=120.0)
        data = s.generate_hours(days * 24)
        bg = data['bg_observed']
        t0 = datetime(2024, 1, 1)
        times = [t0 + timedelta(minutes=5 * i) for i in range(len(bg))]
        stats_d = basic_stats(bg)
        ranges_d = clinical_ranges(bg)
        var_d = variability_metrics(bg, step_min=5)
        diurnal = diurnal_means(times, bg)
        hypo_d = episodes(bg, 70, below=True, step_min=5)
        hyper_d = episodes(bg, 180, below=False, step_min=5)
        rec = {'rid': str(seed),
               **stats_d, **ranges_d, **var_d,
               'hypo_count_per_day': len(hypo_d) / days,
               'hyper_count_per_day': len(hyper_d) / days}
        sim_per.append(rec)
        sim_pooled_bg.append(bg[~np.isnan(bg)])
        sim_diurnals.append(diurnal)
        sim_hypo_durs.extend(hypo_d)
        sim_hyper_durs.extend(hyper_d)
        if (seed + 1) % 10 == 0:
            print(f'  seed {seed + 1}/{n_seeds}')
    sim = {
        'per': sim_per,
        'agg': aggregate(sim_per),
        'pooled_bg': np.concatenate(sim_pooled_bg),
        'diurnal': np.nanmean(np.array(sim_diurnals), axis=0),
        'hypo_durs': sim_hypo_durs,
        'hyper_durs': sim_hyper_durs,
    }
    return ohio, shang, sim


def fig_clinical_ranges(ohio, shang, sim, output):
    """Grouped bar chart: TBR2 / TBR1 / TIR / TAR1 / TAR2 per dataset."""
    labels = ['TBR2 <54', 'TBR1 54-70', 'TIR 70-180', 'TAR1 180-250', 'TAR2 >250']
    keys = ['TBR2_pct', 'TBR1_pct', 'TIR_pct', 'TAR1_pct', 'TAR2_pct']
    ohio_vals = [ohio['agg'][k][0] for k in keys]
    shang_vals = [shang['agg'][k][0] for k in keys]
    sim_vals = [sim['agg'][k][0] for k in keys]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(labels))
    w = 0.27
    bo = ax.bar(x - w, ohio_vals, w, label='OhioT1DM',     color=COL_OHIO,  edgecolor='black', linewidth=0.4)
    bs = ax.bar(x,     shang_vals, w, label='ShanghaiT1DM', color=COL_SHANG, edgecolor='black', linewidth=0.4)
    bm = ax.bar(x + w, sim_vals,  w, label='T1DMSIM',      color=COL_SIM,   edgecolor='black', linewidth=0.4)

    for bars, vals in [(bo, ohio_vals), (bs, shang_vals), (bm, sim_vals)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.6,
                    f'{v:.1f}', ha='center', va='bottom', fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('% of CGM time')
    ax.set_title('Clinical glucose ranges — per-record mean across each dataset',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(0, max(max(ohio_vals), max(shang_vals), max(sim_vals)) * 1.18)
    plt.tight_layout()
    plt.savefig(output, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {output}')


def fig_diurnal(ohio, shang, sim, output):
    """Hour-of-day mean BG curves."""
    hours = np.arange(24)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.axhspan(70, 180, color='lightgreen', alpha=0.25, label='TIR (70–180)')
    ax.plot(hours, ohio['diurnal'],  color=COL_OHIO,  lw=2.4, marker='o', markersize=4, label='OhioT1DM')
    ax.plot(hours, shang['diurnal'], color=COL_SHANG, lw=2.4, marker='s', markersize=4, label='ShanghaiT1DM')
    ax.plot(hours, sim['diurnal'],   color=COL_SIM,   lw=2.4, marker='^', markersize=4, label='T1DMSIM')
    ax.set_xticks(np.arange(0, 25, 3))
    ax.set_xlabel('Hour of day')
    ax.set_ylabel('Mean BG (mg/dL)')
    ax.set_title('Diurnal mean BG by hour-of-day', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, 23.5)
    plt.tight_layout()
    plt.savefig(output, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {output}')


def fig_histogram(ohio, shang, sim, output):
    """Overlapping density histograms of pooled CGM values."""
    bins = np.arange(40, 400, 8)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.axvspan(70, 180, color='lightgreen', alpha=0.2)
    ax.hist(ohio['pooled_bg'],  bins=bins, density=True, histtype='step', linewidth=2.0, color=COL_OHIO,  label=f"OhioT1DM (n={len(ohio['pooled_bg'])})")
    ax.hist(shang['pooled_bg'], bins=bins, density=True, histtype='step', linewidth=2.0, color=COL_SHANG, label=f"ShanghaiT1DM (n={len(shang['pooled_bg'])})")
    ax.hist(sim['pooled_bg'],   bins=bins, density=True, histtype='step', linewidth=2.0, color=COL_SIM,   label=f"T1DMSIM (n={len(sim['pooled_bg'])})")
    ax.axvline(54,  color='red',    alpha=0.5, ls=':', lw=1)
    ax.axvline(70,  color='black',  alpha=0.5, ls=':', lw=1)
    ax.axvline(180, color='black',  alpha=0.5, ls=':', lw=1)
    ax.axvline(250, color='orange', alpha=0.5, ls=':', lw=1)
    ax.set_xlabel('Blood glucose (mg/dL)')
    ax.set_ylabel('Density')
    ax.set_title('Pooled CGM-value distribution', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(40, 400)
    plt.tight_layout()
    plt.savefig(output, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {output}')


def fig_episodes(ohio, shang, sim, output):
    """Pooled hypo + hyper episode duration boxplots."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    hypo_data  = [ohio['hypo_durs'],  shang['hypo_durs'],  sim['hypo_durs']]
    hyper_data = [ohio['hyper_durs'], shang['hyper_durs'], sim['hyper_durs']]
    colors = [COL_OHIO, COL_SHANG, COL_SIM]
    labels = ['OhioT1DM', 'ShanghaiT1DM', 'T1DMSIM']

    for ax, data, title in [(axes[0], hypo_data, 'Hypo episodes (<70 mg/dL)'),
                            (axes[1], hyper_data, 'Hyper episodes (>180 mg/dL)')]:
        bp = ax.boxplot(data, positions=[1, 2, 3], widths=0.55, patch_artist=True,
                        medianprops=dict(color='black', lw=2), showfliers=True,
                        flierprops=dict(marker='.', markersize=3, alpha=0.4))
        for patch, c in zip(bp['boxes'], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.55)
        for i, (d, c) in enumerate(zip(data, colors)):
            ax.text(i + 1, ax.get_ylim()[1] * 0.94, f'n={len(d):,}',
                    ha='center', fontsize=8.5, color='dimgray')
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel('Episode duration (min)')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Episode duration distributions (log scale)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(output, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {output}')


def copy_curve_plots():
    """Copy curve PNGs from /tmp into assets/ with cleaner names."""
    mapping = {
        '/tmp/curves_24h_grid.png':           os.path.join(ASSETS, '24h_traces.png'),
        '/tmp/curves_postmeal_envelope.png':  os.path.join(ASSETS, 'postmeal_envelope.png'),
        '/tmp/curves_quantitative.png':       os.path.join(ASSETS, 'quantitative_shape.png'),
        '/tmp/curves_nocturnal_grid.png':     os.path.join(ASSETS, 'nocturnal_traces.png'),
    }
    for src, dst in mapping.items():
        if os.path.exists(src):
            shutil.copy(src, dst)
            print(f'  copied {src} -> {dst}')
        else:
            print(f'  MISSING (run /tmp/compare_curves.py first): {src}')


def main():
    print('\n=== Running 3-way comparison pipeline ===')
    ohio, shang, sim = run_3way()

    print('\n=== Generating stat figures ===')
    fig_clinical_ranges(ohio, shang, sim, os.path.join(ASSETS, 'clinical_ranges.png'))
    fig_diurnal       (ohio, shang, sim, os.path.join(ASSETS, 'diurnal_bg.png'))
    fig_histogram     (ohio, shang, sim, os.path.join(ASSETS, 'bg_histogram.png'))
    fig_episodes      (ohio, shang, sim, os.path.join(ASSETS, 'episode_durations.png'))

    print('\n=== Copying curve-shape figures ===')
    copy_curve_plots()

    print(f'\nDone. All figures in {ASSETS}/')


if __name__ == '__main__':
    main()
