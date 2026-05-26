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
| **mean** | 162.1 | 164.7 | 170.1 | +8.1 | +5.4 |
| **median** | 155.2 | 156.6 | 151.2 | -4.0 | -5.4 |
| std | 60.8 | 72.3 | 91.1 | +30.3 | +18.8 |
| IQR | 86.2 | 106.2 | 119.2 | +33.0 | +13.0 |
| CV (%) | 37.5 | 43.9 | 53.6 | +16.0 pp | +9.7 pp |
| skewness | 0.58 | 0.51 | 1.13 | +0.54 | +0.61 |
| excess kurtosis | 0.15 | -0.14 | 1.13 | +0.98 | +1.27 |
| min | 40.0 | 39.6 | 22.1 | — | — |
| max | 400.0 | 475.2 | 500.0 | — | — |

### 3.2 Percentiles of the pooled distribution

| Percentile | OhioT1DM | ShanghaiT1DM | T1DMSIM | Sim − Ohio | Sim − Shang |
|---|---:|---:|---:|---:|---:|
| p1 | 57.0 | 41.3 | 54.7 | -2.3 | +13.4 |
| p5 | 76.0 | 61.2 | 63.7 | -12.3 | +2.5 |
| p10 | 88.0 | 75.6 | 72.8 | -15.2 | -2.8 |
| p25 | 115.4 | 108.0 | 98.4 | -17.0 | -9.6 |
| p50 | 155.2 | 156.6 | 151.2 | -4.0 | -5.4 |
| p75 | 201.6 | 214.2 | 217.6 | +16.0 | +3.4 |
| p90 | 244.6 | 264.6 | 297.0 | +52.4 | +32.4 |
| p95 | 271.0 | 291.6 | 350.7 | +79.7 | +59.1 |
| p99 | 325.8 | 349.2 | 466.0 | +140.2 | +116.8 |

![Percentile curves](figures/percentile_curves.png)

![Pooled PDF](figures/pdf_pooled.png)

![Pooled empirical CDF](figures/cdf_pooled.png)

![Q-Q vs Ohio and Shanghai](figures/qq.png)

### 3.3 Distribution-distance statistics

| Pair | KS statistic | KS p-value | Wasserstein-1 (mg/dL) | JS divergence (5 mg/dL bins) |
|---|---:|---:|---:|---:|
| Ohio vs Shanghai | 0.063 | 3.5 × 10⁻⁴⁶ | 10.1 | 0.013 |
| Sim vs Ohio | 0.099 | < 10⁻³⁰⁰ | 21.3 | 0.026 |
| Sim vs Shanghai | 0.058 | 2.0 × 10⁻⁴⁵ | 13.0 | 0.014 |

KS p-values fall to numerical zero in the right tail at these sample sizes
(Ohio ~85k, Sim ~600k); the magnitudes of the KS statistic and the
Wasserstein-1 distance are the meaningful quantities, not p.

---

## 4. Clinical glycemic indices

Per-record means ± std across each cohort.

| Index | OhioT1DM | ShanghaiT1DM | T1DMSIM |
|---|---|---|---|
| GMI / eA1c proxy | 7.19 ± 0.39 | 7.22 ± 0.74 | 7.38 ± 0.77 |
| **LBGI** (low-BG risk) | 0.86 ± 0.49 | 1.82 ± 1.76 | **1.70 ± 0.73** |
| **HBGI** (high-BG risk) | 7.60 ± 2.53 | 8.58 ± 4.28 | **10.65 ± 5.61** |
| J-index | 49.2 ± 9.2 | 52.6 ± 17.1 | 66.4 ± 24.4 |
| M-value (ref 120) | 11.1 ± 3.8 | 15.8 ± 6.2 | 21.9 ± 12.0 |

Pooled (not per-record) risk indices, for reference:

| | Ohio | Shanghai | Sim |
|---|---:|---:|---:|
| LBGI (pooled) | 0.85 | 1.87 | 1.70 |
| HBGI (pooled) | 7.54 | 8.87 | 10.65 |
| J-index (pooled) | 49.7 | 56.2 | 68.2 |
| M-value (pooled) | 11.0 | 16.5 | 21.9 |

### 4.1 Time-in-range, per-record cohort summary

| Range | OhioT1DM | ShanghaiT1DM | T1DMSIM |
|---|---|---|---|
| TBR2 (<54)        | 0.73 ± 0.68 | 2.79 ± 3.77 | 0.83 ± 0.47 |
| TBR1 (54–70)      | 2.57 ± 1.61 | 4.72 ± 3.97 | 7.56 ± 3.38 |
| **TIR (70–180)**  | **60.5 ± 10.2** | **54.7 ± 14.5** | **54.0 ± 12.3** |
| TAR1 (180–250)    | 27.4 ± 6.1 | 25.1 ± 11.7 | 20.2 ± 4.6 |
| TAR2 (>250)       | 8.88 ± 6.11 | 12.64 ± 8.91 | 17.41 ± 11.62 |

![Clinical-range cohort comparison](../assets/clinical_ranges.png)

(The bar chart from the README is included here for direct reference; the
figure is produced by `scripts/generate_comparison_figures.py`.)

---

## 5. Variability and complexity

Per-record mean ± std.

| Metric (native cadence) | OhioT1DM | ShanghaiT1DM | T1DMSIM |
|---|---|---|---|
| CV (%)              | 36.2 ± 4.5   | 38.6 ± 6.8   | **49.6 ± 6.6**   |
| MAGE (mg/dL)        | 103.9 ± 15.4     | 123.4 ± 30.0     | 154.9 ± 19.9     |
| CONGA-1h (mg/dL)    | 39.4 ± 5.6 | 34.2 ± 7.2 | 47.7 ± 4.6 |
| CONGA-4h (mg/dL)    | 76.1 ± 11.4 | 75.1 ± 17.7 | 99.2 ± 12.2 |
| MODD (mg/dL)        | 61.1 ± 8.9     | 53.3 ± 12.8     | **74.0 ± 14.0**     |
| Sample entropy      | 0.87 ± 0.10 | 0.44 ± 0.08¹ | 0.74 ± 0.10 |

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
| 5 min   | 0.995 | (n/a)  | 0.997 |
| 15 min  | 0.969 | 0.984  | 0.982 |
| 30 min  | 0.911 | 0.946  | 0.943 |
| 1 h     | 0.765 | 0.840  | 0.825 |
| 2 h     | 0.484 | 0.606  | 0.563 |
| 4 h     | 0.137 | 0.254  | 0.269 |
| **8 h**     | **-0.004** | **-0.028** | **0.113** |
| **12 h**    | **-0.010** | **-0.050** | **0.099** |
| 24 h    | 0.116 | 0.378  | **0.291** |

![Autocorrelation across lag](figures/acf.png)

### 6.2 Rate-of-change (Δ-BG)

![Δ-BG distribution at native cadence](figures/delta_distribution.png)

Per-record Δ-BG standard deviation (mean across records, native cadence):
Ohio 5.55 mg/dL · Shanghai 10.65 mg/dL ·
Sim 6.36 mg/dL. Shanghai's value is at 15-min cadence and
is not directly comparable to the 5-min values from Ohio and the simulator.

### 6.3 Diurnal pattern (hour-of-day across records)

![Hour-of-day mean with ±1σ envelope](figures/diurnal_envelope.png)

![Hour-of-day median with IQR envelope](figures/diurnal_envelope_median.png)

Hour-by-hour mean BG (mg/dL):

|   | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Ohio | 149 | 151 | 155 | 160 | 164 | 169 | 173 | 179 | 186 | 178 | 164 | 153 | 154 | 161 | 166 | 165 | 163 | 160 | 162 | 162 | 157 | 158 | 156 | 151 |
| Shanghai | 166 | 164 | 163 | 159 | 156 | 158 | 165 | 169 | 192 | 175 | 144 | 137 | 149 | 143 | 147 | 157 | 166 | 179 | 184 | 175 | 170 | 169 | 168 | 167 |
| Sim | 156 | 137 | 127 | 122 | 123 | 128 | 138 | 151 | 169 | 186 | 193 | 189 | 182 | 182 | 188 | 192 | 187 | 175 | 170 | 182 | 203 | 213 | 206 | 183 |

Hour-by-hour median BG (mg/dL):

|   | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Ohio | 144 | 137 | 139 | 139 | 142 | 157 | 160 | 166 | 180 | 164 | 147 | 148 | 152 | 162 | 163 | 160 | 159 | 155 | 156 | 160 | 154 | 149 | 143 | 148 |
| Shanghai | 156 | 159 | 152 | 144 | 145 | 144 | 150 | 161 | 192 | 164 | 126 | 138 | 149 | 135 | 147 | 165 | 158 | 176 | 199 | 170 | 170 | 175 | 171 | 166 |
| Sim | 136 | 110 | 93 | 94 | 93 | 100 | 110 | 122 | 146 | 167 | 179 | 176 | 162 | 158 | 163 | 172 | 165 | 151 | 146 | 162 | 184 | 198 | 185 | 165 |

![Weekday × hour mean heatmap](figures/weekday_heatmap.png)

![Weekday × hour median heatmap](figures/weekday_heatmap_median.png)

---

## 7. Excursion-level dynamics

### 7.1 Episode counts and durations

Per-record means ± std.

| Metric | OhioT1DM | ShanghaiT1DM | T1DMSIM |
|---|---|---|---|
| Hypo (<70) episodes / day      | 0.81 ± 0.40 | 1.02 ± 0.74 | **1.99 ± 0.87** |
| Severe-hypo (<54) eps / day   | 0.20 ± 0.20 | 0.51 ± 0.47 | 0.31 ± 0.18 |
| Hyper (>180) episodes / day   | 2.61 ± 0.26 | 1.87 ± 0.71 | 2.02 ± 0.29 |
| Severe-hyper (>250) eps / day | 1.06 ± 0.38 | 1.12 ± 0.68 | 1.31 ± 0.50 |
| Hypo median duration (min)    | 33.3 | 69.4 | 48.7 |
| Hypo p90 duration (min)       | 89.8 | 179.2 | **100.5** |
| Hyper median duration (min)   | 131.2 | 213.3 | 185.3 |
| Hyper p90 duration (min)      | 422.8 | 622.6 | **561.0** |

![Episode duration boxplots](figures/episode_durations.png)

### 7.2 Hypo recovery time

![Hypo recovery time from BG<70 to BG≥80](figures/recovery_time.png)

Time from the first sub-70 sample to the next ≥ 80 sample:

| Cohort | n events | median (min) | p75 (min) | p90 (min) | p99 (min) | max (min) |
|---|---:|---:|---:|---:|---:|---:|
| Ohio     |   284 | 50 | 81 | 134 | 216 | 295 |
| Shanghai |   157 | 90 | 195 | 300 | 510 | 555 |
| Sim      | 3,590 | 70 | 120 | 190 | 371 | 580 |

---

## 8. Per-record (per-patient) heterogeneity

![Per-record TIR vs TBR scatter](figures/per_patient_scatter.png)

| Cohort | TIR IQR (pp) | TIR min–max (pp) | Mean-BG std across records |
|---|---:|---|---:|
| Ohio     | 9.3 | 40.3 – 71.7 | 16.2 |
| Shanghai | 23.1 | 32.1 – 77.3 | 31.0 |
| Sim      | 11.7 | 17.4 – 72.6 | 32.3 |

![LBGI and HBGI per-record boxplots](figures/risk_indices.png)

---

## 9. Side-by-side summary

Raw deltas only — no qualitative verdicts. See sections 3–8 for context.

| Quantity | T1DMSIM | OhioT1DM | ShanghaiT1DM | Sim − Ohio | Sim − Shang |
|---|---:|---:|---:|---:|---:|
| Pooled mean BG (mg/dL) | 170.1 | 162.1 | 164.7 | +8.1 | +5.4 |
| Pooled median BG (mg/dL) | 151.2 | 155.2 | 156.6 | -4.0 | -5.4 |
| Pooled std (mg/dL) | 91.1 | 60.8 | 72.3 | +30.3 | +18.8 |
| Pooled CV (%) | 53.6 | 37.5 | 43.9 | +16.0 | +9.7 |
| Pooled skewness | 1.13 | 0.58 | 0.51 | +0.54 | +0.61 |
| Pooled excess kurtosis | 1.13 | 0.15 | -0.14 | +0.98 | +1.27 |
| Pooled p99 (mg/dL) | 466.0 | 325.8 | 349.2 | +140.2 | +116.8 |
| GMI (per-record mean) | 7.38 | 7.19 | 7.22 | +0.19 | +0.16 |
| LBGI (per-record mean) | 1.70 | 0.86 | 1.82 | +0.85 | -0.12 |
| HBGI (per-record mean) | 10.65 | 7.60 | 8.58 | +3.06 | +2.08 |
| TIR % (per-record mean) | 54.0 | 60.5 | 54.7 | -6.5 | -0.8 |
| TBR1 % (per-record mean) | 7.56 | 2.57 | 4.72 | +4.99 | +2.84 |
| TBR2 % (per-record mean) | 0.83 | 0.73 | 2.79 | +0.09 | -1.96 |
| TAR1 % (per-record mean) | 20.2 | 27.4 | 25.1 | -7.1 | -4.9 |
| TAR2 % (per-record mean) | 17.4 | 8.9 | 12.6 | +8.5 | +4.8 |
| MAGE (mg/dL) | 154.9 | 103.9 | 123.4 | +51.0 | +31.5 |
| CONGA-1h (mg/dL) | 47.7 | 39.4 | 34.2 | +8.3 | +13.6 |
| CONGA-4h (mg/dL) | 99.2 | 76.1 | 75.1 | +23.1 | +24.1 |
| MODD (mg/dL) | 74.0 | 61.1 | 53.3 | +12.9 | +20.7 |
| Hypo episodes / day | 1.99 | 0.81 | 1.02 | +1.18 | +0.97 |
| Severe-hypo eps / day | 0.31 | 0.20 | 0.51 | +0.11 | -0.20 |
| Hyper episodes / day | 2.02 | 2.61 | 1.87 | -0.59 | +0.15 |
| Severe-hyper eps / day | 1.31 | 1.06 | 1.12 | +0.24 | +0.19 |
| Hypo p90 duration (min) | 100.5 | 89.8 | 179.2 | +10.7 | -78.8 |
| Hyper p90 duration (min) | 561.0 | 422.8 | 622.6 | +138.2 | -61.6 |
| Hypo recovery median (min) | 70.0 | 50.0 | 90.0 | +20.0 | -20.0 |
| Wasserstein-1 vs Ohio (mg/dL) | 21.3 | 10.1 | 13.0 | +11.1 | +8.3 |
| KS statistic vs Ohio | 0.099 | 0.063 | 0.058 | +0.036 | +0.041 |

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
