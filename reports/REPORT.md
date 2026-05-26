# T1DMSIM vs OhioT1DM vs ShanghaiT1DM — Statistical Comparison Report

Comprehensive statistical comparison of the synthetic blood-glucose traces
produced by `simulator.py` against two non-redistributable real-world CGM
corpora. Goes well beyond the summary panel in `README.md`: full
percentile tables, distribution-distance statistics (Kolmogorov–Smirnov,
Wasserstein, Jensen–Shannon), Kovatchev risk indices, MAGE / CONGA / MODD /
sample entropy, autocorrelation across nine lags, rate-of-change distributions,
hour-of-day envelopes, weekday × hour heatmaps, per-record TIR/TBR scatter, and
expanded excursion-level metrics.

This file is regenerated end-to-end by `reports/build_report.py`. Raw stats
are persisted to `reports/stats.json`; figures live in `reports/figures/`.

---

## 1. Corpora at a glance

| Dataset | Records | Cadence | Total CGM-days | Cohort | Notes |
|---|---:|---:|---:|---|---|
| OhioT1DM | 6 records (file pairs) | 5 min Dexcom | 321.7 | US adults, pump + announced meals | training + testing periods concatenated per patient |
| ShanghaiT1DM | 16 records | **15 min** | 163.5 | CN adults, mixed CSII + MDI (incl. regular Novolin R), BMI ≈ 21 | shorter individual records (~10 d) |
| T1DMSIM | 30 seeds × 70 days | 5 min | 2100.0 | synthetic, seeds 0–29, 24 h warm-up discarded | `initial_bg = 120 mg/dL`, `bg_observed` (sensor-noised) |

Both real datasets are gitignored. The simulator is exercised as in
`scripts/compare_all_datasets.py`: 24 h warm-up to clear the `initial_bg = 120`
transient, then the next 70 days are captured.

---

## 2. Methodology

- **Resampling.** Ohio CGM is irregular Dexcom samples; it is linearly
  interpolated onto a 5 min grid with gaps > 30 min NaN-bridged. Shanghai is
  similarly resampled to a 15 min grid with > 60 min gaps as NaN. The simulator
  is already on a 5 min grid. All statistics ignore NaN.
- **Cadence-aware comparison.** Rate-of-change Δ-BG, ACF, CONGA, MAGE, and MODD
  are computed at each cohort's **native** cadence. Cross-cadence comparison is
  flagged where it materially affects interpretation (notably Δ-BG std and
  sample entropy).
- **Distribution distances.** KS statistic, KS p-value, Wasserstein-1 distance,
  and Jensen–Shannon divergence (5 mg/dL bins, 40–400 mg/dL) computed on the
  pooled per-cohort CGM-value vector.
- **Risk indices.** LBGI / HBGI per Kovatchev (1997), J-index = 10⁻³·(μ+σ)²,
  M-value with reference 120 mg/dL.
- **MAGE.** Mean amplitude of peak-trough swings exceeding 1·σ_BG.
- **CONGA-h.** Standard deviation of `bg[t+h] − bg[t]`.
- **MODD.** Mean of `|bg[t+24h] − bg[t]|`.
- **Sample entropy.** SampEn(m=2, r=0.2·σ); to bound cost on long traces,
  records are uniformly subsampled to 2,500 points with a fixed RNG seed (0)
  before computation.
- **Episodes.** Contiguous runs across a threshold lasting ≥ 15 min, NaN
  gaps treated as in-range so a sensor dropout does not split an episode.

---

## 3. Headline numbers

### 3.1 Pooled central moments

| Metric (mg/dL) | OhioT1DM | ShanghaiT1DM | T1DMSIM | Sim − Ohio | Sim − Shang |
|---|---:|---:|---:|---:|---:|
| n (samples) | 85,295 | 15,696 | 604,800 | — | — |
| **mean** | 162.1 | 164.7 | 181.0 | +18.9 | +16.2 |
| **median** | 155.2 | 156.6 | 162.0 | +6.8 | +5.4 |
| std | 60.8 | 72.3 | 88.7 | +27.9 | +16.4 |
| IQR | 86.2 | 106.2 | 119.3 | +33.1 | +13.1 |
| CV (%) | 37.5 | 43.9 | 49.0 | +11.5 pp | +5.1 pp |
| skewness | 0.58 | 0.51 | 0.95 | +0.37 | +0.44 |
| excess kurtosis | 0.15 | -0.14 | 0.65 | +0.50 | +0.79 |
| min | 40.0 | 39.6 | 23.1 | — | — |
| max | 400.0 | 475.2 | 500.0 | — | — |

### 3.2 Percentiles of the pooled distribution

| Percentile | OhioT1DM | ShanghaiT1DM | T1DMSIM | Sim − Ohio | Sim − Shang |
|---|---:|---:|---:|---:|---:|
| p1 | 57.0 | 41.3 | 55.3 | -1.7 | +14.0 |
| p5 | 76.0 | 61.2 | 69.5 | -6.5 | +8.3 |
| p10 | 88.0 | 75.6 | 82.6 | -5.4 | +7.0 |
| p25 | 115.4 | 108.0 | 113.1 | -2.3 | +5.1 |
| p50 | 155.2 | 156.6 | 162.0 | +6.8 | +5.4 |
| p75 | 201.6 | 214.2 | 232.4 | +30.8 | +18.2 |
| p90 | 244.6 | 264.6 | 306.8 | +62.2 | +42.2 |
| p95 | 271.0 | 291.6 | 350.8 | +79.8 | +59.2 |
| p99 | 325.8 | 349.2 | 452.4 | +126.6 | +103.2 |

![Percentile curves](figures/percentile_curves.png)

![Pooled PDF](figures/pdf_pooled.png)

![Pooled empirical CDF](figures/cdf_pooled.png)

![Q-Q vs Ohio and Shanghai](figures/qq.png)

### 3.3 Distribution-distance statistics

| Pair | KS statistic | KS p-value | Wasserstein-1 (mg/dL) | JS divergence (5 mg/dL bins) |
|---|---:|---:|---:|---:|
| Ohio vs Shanghai | 0.063 | 3.5 × 10⁻⁴⁶ | 10.1 | 0.013 |
| Sim vs Ohio | 0.119 | < 10⁻³⁰⁰ | 21.3 | 0.020 |
| Sim vs Shanghai | 0.078 | 7.6 × 10⁻⁸¹ | 16.2 | 0.012 |

KS p-values fall to numerical zero in the right tail at these sample sizes
(Ohio ~85k, Sim ~600k); the magnitudes of the KS statistic and the
Wasserstein-1 distance are the meaningful quantities, not p.

---

## 4. Clinical glycemic indices

Per-record means ± std across each cohort.

| Index | OhioT1DM | ShanghaiT1DM | T1DMSIM |
|---|---|---|---|
| GMI / eA1c proxy | 7.19 ± 0.39 | 7.22 ± 0.74 | 7.64 ± 0.69 |
| **LBGI** (low-BG risk) | 0.86 ± 0.49 | 1.82 ± 1.76 | **1.13 ± 0.41** |
| **HBGI** (high-BG risk) | 7.60 ± 2.53 | 8.58 ± 4.28 | **11.92 ± 5.13** |
| J-index | 49.2 ± 9.2 | 52.6 ± 17.1 | 71.1 ± 22.1 |
| M-value (ref 120) | 11.1 ± 3.8 | 15.8 ± 6.2 | 22.7 ± 10.9 |

Pooled (not per-record) risk indices, for reference:

| | Ohio | Shanghai | Sim |
|---|---:|---:|---:|
| LBGI (pooled) | 0.85 | 1.87 | 1.13 |
| HBGI (pooled) | 7.54 | 8.87 | 11.92 |
| J-index (pooled) | 49.7 | 56.2 | 72.7 |
| M-value (pooled) | 11.0 | 16.5 | 22.7 |

### 4.1 Time-in-range, per-record cohort summary

| Range | OhioT1DM | ShanghaiT1DM | T1DMSIM |
|---|---|---|---|
| TBR2 (<54)        | 0.73 ± 0.68 | 2.79 ± 3.77 | 0.79 ± 0.38 |
| TBR1 (54–70)      | 2.57 ± 1.61 | 4.72 ± 3.97 | 4.40 ± 1.68 |
| **TIR (70–180)**  | **60.5 ± 10.2** | **54.7 ± 14.5** | **52.4 ± 12.4** |
| TAR1 (180–250)    | 27.4 ± 6.1 | 25.1 ± 11.7 | 21.9 ± 5.1 |
| TAR2 (>250)       | 8.88 ± 6.11 | 12.64 ± 8.91 | 20.54 ± 11.18 |

![Clinical-range cohort comparison](../assets/clinical_ranges.png)

(The bar chart from the README is included here for direct reference; the
figure is produced by `scripts/generate_comparison_figures.py`.)

---

## 5. Variability and complexity

Per-record mean ± std.

| Metric (native cadence) | OhioT1DM | ShanghaiT1DM | T1DMSIM |
|---|---|---|---|
| CV (%)              | 36.2 ± 4.5   | 38.6 ± 6.8   | **45.9 ± 4.9**   |
| MAGE (mg/dL)        | 103.9 ± 15.4     | 123.4 ± 30.0     | 154.2 ± 22.0     |
| CONGA-1h (mg/dL)    | 39.4 ± 5.6 | 34.2 ± 7.2 | 52.3 ± 5.5 |
| CONGA-4h (mg/dL)    | 76.1 ± 11.4 | 75.1 ± 17.7 | 99.2 ± 11.3 |
| MODD (mg/dL)        | 61.1 ± 8.9     | 53.3 ± 12.8     | **82.0 ± 17.5**     |
| Sample entropy      | 0.87 ± 0.10 | 0.44 ± 0.08¹ | 0.92 ± 0.07 |

¹ Shanghai SampEn is computed on 15-min samples, which collapses the
  fine-scale jitter that drives SampEn at 5 min — the lower value is mostly a
  cadence artefact, not a real complexity difference.

![Variability and complexity panel](figures/variability_metrics.png)

---

## 6. Temporal dynamics

### 6.1 Autocorrelation

Pooled (mean across records) Pearson autocorrelation at the indicated lag.

| Lag         | OhioT1DM | ShanghaiT1DM | T1DMSIM |
|---|---|---|---|
| 5 min   | 0.995 | (n/a)  | 0.996 |
| 15 min  | 0.969 | 0.984  | 0.979 |
| 30 min  | 0.911 | 0.946  | 0.930 |
| 1 h     | 0.765 | 0.840  | 0.788 |
| 2 h     | 0.484 | 0.606  | 0.489 |
| 4 h     | 0.137 | 0.254  | 0.254 |
| **8 h**     | **-0.004** | **-0.028** | **0.207** |
| **12 h**    | **-0.010** | **-0.050** | **0.238** |
| 24 h    | 0.116 | 0.378  | **0.161** |

![Autocorrelation across lag](figures/acf.png)

### 6.2 Rate-of-change (Δ-BG)

![Δ-BG distribution at native cadence](figures/delta_distribution.png)

Per-record Δ-BG standard deviation (mean across records, native cadence):
Ohio 5.55 mg/dL · Shanghai 10.65 mg/dL ·
Sim 6.89 mg/dL. Shanghai's value is at 15-min cadence and
is not directly comparable to the 5-min values from Ohio and the simulator.

### 6.3 Diurnal pattern (hour-of-day mean ± 1σ across records)

![Hour-of-day mean with ±1σ envelope](figures/diurnal_envelope.png)

Hour-by-hour mean BG (mg/dL):

|   | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Ohio | 149 | 151 | 155 | 160 | 164 | 169 | 173 | 179 | 186 | 178 | 164 | 153 | 154 | 161 | 166 | 165 | 163 | 160 | 162 | 162 | 157 | 158 | 156 | 151 |
| Shanghai | 166 | 164 | 163 | 159 | 156 | 158 | 165 | 169 | 192 | 175 | 144 | 137 | 149 | 143 | 147 | 157 | 166 | 179 | 184 | 175 | 170 | 169 | 168 | 167 |
| Sim | 166 | 155 | 154 | 158 | 167 | 179 | 197 | 216 | 228 | 223 | 200 | 172 | 157 | 161 | 173 | 175 | 166 | 158 | 161 | 177 | 201 | 211 | 204 | 186 |

![Weekday × hour heatmap](figures/weekday_heatmap.png)

---

## 7. Excursion-level dynamics

### 7.1 Episode counts and durations

Per-record means ± std.

| Metric | OhioT1DM | ShanghaiT1DM | T1DMSIM |
|---|---|---|---|
| Hypo (<70) episodes / day      | 0.81 ± 0.40 | 1.02 ± 0.74 | **1.92 ± 0.74** |
| Severe-hypo (<54) eps / day   | 0.20 ± 0.20 | 0.51 ± 0.47 | 0.26 ± 0.17 |
| Hyper (>180) episodes / day   | 2.61 ± 0.26 | 1.87 ± 0.71 | 2.13 ± 0.30 |
| Severe-hyper (>250) eps / day | 1.06 ± 0.38 | 1.12 ± 0.68 | 1.61 ± 0.50 |
| Hypo median duration (min)    | 33.3 | 69.4 | 32.0 |
| Hypo p90 duration (min)       | 89.8 | 179.2 | **56.7** |
| Hyper median duration (min)   | 131.2 | 213.3 | 174.4 |
| Hyper p90 duration (min)      | 422.8 | 622.6 | **667.0** |

![Episode duration boxplots](figures/episode_durations.png)

### 7.2 Hypo recovery time

![Hypo recovery time from BG<70 to BG≥80](figures/recovery_time.png)

Time from the first sub-70 sample to the next ≥ 80 sample:

| Cohort | n events | median (min) | p75 (min) | p90 (min) | p99 (min) | max (min) |
|---|---:|---:|---:|---:|---:|---:|
| Ohio     |   284 | 50 | 81 | 134 | 216 | 295 |
| Shanghai |   157 | 90 | 195 | 300 | 510 | 555 |
| Sim      | 4,193 | 40 | 50 | 75 | 210 | 400 |

---

## 8. Per-record (per-patient) heterogeneity

![Per-record TIR vs TBR scatter](figures/per_patient_scatter.png)

| Cohort | TIR IQR (pp) | TIR min–max (pp) | Mean-BG std across records |
|---|---:|---|---:|
| Ohio     | 9.3 | 40.3 – 71.7 | 16.2 |
| Shanghai | 23.1 | 32.1 – 77.3 | 31.0 |
| Sim      | 17.0 | 24.1 – 68.4 | 28.8 |

![LBGI and HBGI per-record boxplots](figures/risk_indices.png)

---

## 9. Side-by-side summary

Raw deltas only — no qualitative verdicts. See sections 3–8 for context.

| Quantity | T1DMSIM | OhioT1DM | ShanghaiT1DM | Sim − Ohio | Sim − Shang |
|---|---:|---:|---:|---:|---:|
| Pooled mean BG (mg/dL) | 181.0 | 162.1 | 164.7 | +18.9 | +16.2 |
| Pooled median BG (mg/dL) | 162.0 | 155.2 | 156.6 | +6.8 | +5.4 |
| Pooled std (mg/dL) | 88.7 | 60.8 | 72.3 | +27.9 | +16.4 |
| Pooled CV (%) | 49.0 | 37.5 | 43.9 | +11.5 | +5.1 |
| Pooled skewness | 0.95 | 0.58 | 0.51 | +0.37 | +0.44 |
| Pooled excess kurtosis | 0.65 | 0.15 | -0.14 | +0.50 | +0.79 |
| Pooled p99 (mg/dL) | 452.4 | 325.8 | 349.2 | +126.6 | +103.2 |
| GMI (per-record mean) | 7.64 | 7.19 | 7.22 | +0.45 | +0.41 |
| LBGI (per-record mean) | 1.13 | 0.86 | 1.82 | +0.27 | -0.69 |
| HBGI (per-record mean) | 11.92 | 7.60 | 8.58 | +4.32 | +3.34 |
| TIR % (per-record mean) | 52.4 | 60.5 | 54.7 | -8.1 | -2.4 |
| TBR1 % (per-record mean) | 4.40 | 2.57 | 4.72 | +1.83 | -0.32 |
| TBR2 % (per-record mean) | 0.79 | 0.73 | 2.79 | +0.05 | -2.00 |
| TAR1 % (per-record mean) | 21.9 | 27.4 | 25.1 | -5.4 | -3.2 |
| TAR2 % (per-record mean) | 20.5 | 8.9 | 12.6 | +11.7 | +7.9 |
| MAGE (mg/dL) | 154.2 | 103.9 | 123.4 | +50.3 | +30.8 |
| CONGA-1h (mg/dL) | 52.3 | 39.4 | 34.2 | +12.9 | +18.1 |
| CONGA-4h (mg/dL) | 99.2 | 76.1 | 75.1 | +23.1 | +24.1 |
| MODD (mg/dL) | 82.0 | 61.1 | 53.3 | +20.9 | +28.8 |
| Hypo episodes / day | 1.92 | 0.81 | 1.02 | +1.12 | +0.90 |
| Severe-hypo eps / day | 0.26 | 0.20 | 0.51 | +0.06 | -0.25 |
| Hyper episodes / day | 2.13 | 2.61 | 1.87 | -0.48 | +0.26 |
| Severe-hyper eps / day | 1.61 | 1.06 | 1.12 | +0.55 | +0.49 |
| Hypo p90 duration (min) | 56.7 | 89.8 | 179.2 | -33.1 | -122.5 |
| Hyper p90 duration (min) | 667.0 | 422.8 | 622.6 | +244.2 | +44.4 |
| Hypo recovery median (min) | 40.0 | 50.0 | 90.0 | -10.0 | -50.0 |
| Wasserstein-1 vs Ohio (mg/dL) | 21.3 | 10.1 | 16.2 | +11.2 | +5.1 |
| KS statistic vs Ohio | 0.119 | 0.063 | 0.078 | +0.056 | +0.042 |

---

## 10. Limitations of this comparison

- **Cohort size.** Ohio (n = 6) and Shanghai (n = 16) are small enough that
  cohort means have non-trivial sampling error; the "real" distribution should
  be taken as a band, not a point. With 30 simulator seeds the sim cohort is
  intentionally larger to bound its own sampling error tightly.
- **Cadence asymmetry.** Shanghai's 15-min cadence collapses Δ-BG std and
  sample entropy relative to 5-min cohorts. Cross-cadence ACF below 30 min is
  not directly comparable.
- **No glucose-controller benchmark.** The simulator output is compared to two
  real human cohorts but not to UVA/Padova `simglucose` here.
- **No external behaviour event matching.** Meal and bolus event distributions
  exist in both Ohio XML and Shanghai sheets, but this report compares CGM
  output only — not the carb-bolus pairing distribution, time-to-meal-peak
  alignment, or exercise/sleep co-occurrence.
- **Sample entropy subsampling.** Records longer than 2,500 points are
  subsampled with `np.random.default_rng(0)` so the metric is reproducible but
  is a Monte-Carlo estimate, not the exact value over the full trace.

---

## 11. Reproduction

```bash
# regenerates reports/stats.json + reports/REPORT.md + reports/figures/*.png
python reports/build_report.py
```

`scripts/compare_all_datasets.py` is reused for the dataset loaders and grid
regularisation. `OhioT1DM/` and `ShanghaiT1DM/` must be placed at the repo root
(both gitignored, both subject to data-use agreements).

Numbers in this file come from one run of `build_report.py`
(30 seeds, 70 days each, 24 h warm-up discarded). Re-running reproduces
them exactly because the simulator is seed-deterministic and the real-data
side is fixed.
