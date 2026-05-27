# T1DMSIM vs OhioT1DM, ShanghaiT1DM, AZT1D — Statistical Comparison Report

`simulator.py` is the seed-driven T1D behaviour simulator described in the
[project README](../README.md). It produces synthetic blood-glucose traces
together with the underlying carb / insulin / exercise / sensitivity factor
curves that generated them. This document compares those synthetic traces
against three real-world CGM corpora — OhioT1DM, ShanghaiT1DM, and AZT1D
(see [References](../README.md#references) in the project README) — across
distributional moments, KS / Wasserstein / JS distances, Kovatchev risk
indices, MAGE / CONGA / MODD / sample entropy, autocorrelation across nine
lags, rate-of-change distributions, hour-of-day envelopes, weekday × hour
heatmaps, per-record TIR / TBR scatter, expanded excursion-level metrics,
and — using AZT1D's richer pump log — a head-to-head comparison of basal-rate
distribution, bolus-event counts, per-meal carb size, and device-mode time
share.

**Cohort sizing.** The simulator is exercised here at 30 seeds × 70
days to give it a sample count comparable to the largest real corpus.
**This is not a fixed corpus** — `T1DMSimulator.generate_hours()` is a
deterministic generator (same seed → same trace), and an arbitrary number of
new patient seeds can be sampled, each producing an arbitrarily long trace.
Every per-record number below ("samples", "CGM-days", "n events", etc.) is a
property of *this particular run*, not an upper bound on what the simulator
can produce.

This file is regenerated end-to-end by `diff/build_report.py`. Raw stats are
persisted to `diff/stats.json`; figures live in `diff/figures/`.

## Table of contents

- [0. Machine-learning summary](#0-machine-learning-summary)
- [1. Corpora at a glance](#1-corpora-at-a-glance)
- [2. Methodology](#2-methodology)
- [3. Headline numbers](#3-headline-numbers)
- [4. Clinical glycemic indices](#4-clinical-glycemic-indices)
- [5. Variability and complexity](#5-variability-and-complexity)
- [6. Temporal dynamics](#6-temporal-dynamics)
- [7. Excursion-level dynamics](#7-excursion-level-dynamics)
- [8. Per-record heterogeneity](#8-per-record-per-patient-heterogeneity)
- [9. AZT1D insulin / carb panel](#9-azt1d-insulin--carb-behaviour-panel)
- [10. Side-by-side summary](#10-side-by-side-summary)
- [11. Limitations of this comparison](#11-limitations-of-this-comparison)
- [12. Reproduction](#12-reproduction)

---

## 0. Machine-learning summary

Stats and visuals tailored to designing an ML pipeline against this data.
The simulator emits a continuous, fully-labelled multivariate stream
(BG, carb intake, basal/bolus insulin, exercise, sensitivity, hepatic output
…) suitable for sequence modelling with regression or class-imbalanced
classification heads. **Volume is not a constraint:** each seed produces
an arbitrarily long trace, and there is no upper bound on how many seeds you
can sample — the figures below describe one particular generation run, not a
fixed corpus size.

### 0.1 Data volume per cohort

| Cohort | Records | Samples | Hours | CGM-days | Cadence |
|---|---:|---:|---:|---:|---:|
| OhioT1DM     |   6 | 85,295 |   7,720 |     322 | 5 min |
| ShanghaiT1DM |  16 | 15,696 |   3,924 |     164 | 15 min |
| AZT1D        |  24 | 288,085 |  24,375 |   1,016 | 5 min |
| **T1DMSIM** *(this run)* | **30** | **604,800** | ** 50,400** | **  2,100** | **5 min** |

The T1DMSIM row above reflects the 30-seed × 70-day run used
to compute every other number in this report. A fresh `build_report.py` call
with `--n-seeds N --days D` (or the equivalent edit in `main()`) scales the
synthetic corpus linearly with no quality penalty — useful when an ML model
needs orders of magnitude more samples than the real corpora can supply.
Relative to that one run, T1DMSIM provides **7.1× the sample count of OhioT1DM**,
**38.5× ShanghaiT1DM**, and
**2.1× AZT1D**.

### 0.2 Normalization statistics

Per-cohort statistics on the pooled BG vector. Use these for input/output
standardization. For neural-net-friendly z-scoring on the simulator output:
`bg_z = (bg - 160.4) / 76.0`. For robust scaling
that is resistant to extreme-hyper outliers: `bg_robust = (bg - 148.9) / 99.3`
(median / IQR).

| Stat (mg/dL) | Ohio | Shanghai | AZT1D | **Sim** |
|---|---:|---:|---:|---:|
| mean    | 162.1  | 164.7  | 148.2  | **160.4**  |
| median  | 155.2  | 156.6  | 139.2  | **148.9**  |
| std     | 60.8   | 72.3   | 47.5   | **76.0**   |
| IQR     | 86.2   | 106.2   | 58.0   | **99.3**   |
| p1      | 57.0    | 41.3    | 67.1    | **55.2**    |
| p99     | 325.8   | 349.2   | 294.8   | **395.3**   |
| min     | 40.0   | 39.6   | 40.0   | **22.1**   |
| max     | 400.0   | 475.2   | 400.0   | **500.0**   |

### 0.3 Sample-level class balance

For classification heads predicting BG-band membership. Percentages are
per-record means across each cohort.

![Class balance per cohort](figures/class_balance.png)

| Band | Threshold | Ohio % | Shang % | AZT1D % | **Sim %** |
|---|---|---:|---:|---:|---:|
| TBR2 | <54     |  0.73 |  2.79 |  0.23 | ** 0.73** |
| TBR1 | 54-70   |  2.57 |  4.72 |  1.03 | ** 6.63** |
| TIR  | 70-180  |  60.5  |  54.7  |  77.1  | ** 59.1**  |
| TAR1 | 180-250 |  27.4 |  25.1 |  17.7 | ** 21.3** |
| TAR2 | >250    |   8.9 |  12.6 |   3.9 | ** 12.2** |

T1DMSIM is intentionally tuned for *elevated mild-hypo (TBR1) and severe-hyper
(TAR2) density* relative to OhioT1DM — the shape of those events (durations,
depths, recovery profiles) still matches real cohorts (see §7), but the rate is
higher to give a classifier more positive examples of each rare-event class
per epoch.

### 0.4 Episode-level event counts

For rare-event detection training (e.g. "will hypo in next N minutes" binary
heads). Each row is a contiguous excursion ≥ 15 min.

| Event class | Ohio | Shanghai | AZT1D | **Sim** |
|---|---:|---:|---:|---:|
| Hypo (<70) episodes        | 261   | 169   | 475   | **3,787** |
| Severe-hypo (<54) episodes | 64    | 78    | 95    | **591** |
| Hyper (>180) episodes      | 840  | 305  | 2,713  | **4,361** |
| Severe-hyper (>250) episodes | 338  | 192  | 619  | **2,282** |

### 0.5 Effective context window

Pooled Pearson autocorrelation decays as the lag grows. The lag at which the
ACF drops below a chosen threshold is a useful order-of-magnitude estimate of
how long an autoregressive model needs to look back.

| ACF threshold | Ohio | Shanghai | AZT1D | **Sim** |
|---|---:|---:|---:|---:|
| 0.5 (50% retained) | 1.9 h | 2.6 h | 1.3 h | **2.7 h** |
| 0.2 (20% retained) | 3.6 h | 4.8 h | 2.4 h | **6.8 h** |

A 4-8h context window covers the meaningful autoregressive signal; longer
contexts add little beyond the half-day BG ACF tail visible in §6.1.

### 0.6 Cross-record heterogeneity

For train/val/test split design. Between-patient variance dominating
within-patient variance means *patient-stratified* splits are required —
otherwise a model trained on one patient set will fail to generalize to
unseen patients. The simulator's between/within ratio is intentionally close
to the real cohorts' so the same split strategy carries over.

| Cohort | Between-patient mean-BG std | Within-patient BG std | Ratio |
|---|---:|---:|---:|
| Ohio     | 16.2 | 58.5 | 0.28 |
| Shanghai | 31.0 | 62.2 | 0.50 |
| AZT1D    | 14.1 | 44.7 | 0.31 |
| **Sim**  | **23.7** | **70.7** | **0.33** |

### 0.7 Diurnal shape (clean line overlay)

The diurnal envelope figure in §6.3 includes ±1σ bands that visually overlap
across cohorts. This clean line overlay isolates the shape comparison.

![Diurnal BG curves — clean line overlay](figures/diurnal_lines.png)

### 0.8 Sim-vs-real domain gap

For models trained on the simulator and evaluated on real CGM, the
Wasserstein-1 distance between the sim and real pooled distributions is a
direct measure of the domain gap that needs to close at inference time.

| Pair | KS | Wasserstein-1 (mg/dL) | JS divergence |
|---|---:|---:|---:|
| **Sim vs Ohio**     | 0.082 | 11.9 | 0.014 |
| **Sim vs Shanghai** | 0.061 | 9.1 | 0.010 |
| **Sim vs AZT1D**    | 0.127 | 23.7 | 0.049 |
| Ohio vs Shanghai (real-vs-real baseline) | 0.063 | 10.1 | 0.013 |
| Ohio vs AZT1D    (real-vs-real baseline) | 0.149 | 17.3 | 0.025 |
| Shanghai vs AZT1D (real-vs-real baseline) | 0.182 | 26.3 | 0.056 |

The smallest Sim-vs-real Wasserstein-1 of 9.1 mg/dL is
*smaller* than the
mean real-vs-real baseline of 17.9 mg/dL — meaning the
simulator's pooled BG distribution is
closer to one real cohort than the three real cohorts are to each other on average.


---

## 1. Corpora at a glance

| Dataset | Records | Cadence | Total CGM-days | Cohort | Notes |
|---|---:|---:|---:|---|---|
| OhioT1DM | 6 records (file pairs) | 5 min Dexcom | 321.7 | US adults, pump + announced meals | training + testing periods concatenated per patient |
| ShanghaiT1DM | 16 records | **15 min** | 163.5 | CN adults, mixed CSII + MDI (incl. regular Novolin R), BMI ≈ 21 | shorter individual records (~10 d) |
| AZT1D | 24 subjects | 5 min Dexcom G6 | 1015.6 | US adults, Mayo Clinic AZ, all on Tandem t:slim X2 Control-IQ (AID) | rich pump event log: bolus type, basal rate, carbs, device mode |
| T1DMSIM *(this run)* | 30 seeds × 70 days | 5 min | 2100.0 | synthetic, seeds 0–29, 24 h warm-up discarded | `initial_bg = 120 mg/dL`, `bg_observed` (sensor-noised); generator is unbounded |

All three real datasets are gitignored. The simulator is exercised as in
`scripts/compare_all_datasets.py`: 24 h warm-up to clear the `initial_bg = 120`
transient, then the next 70 days are captured.

---

## 2. Methodology

- **Resampling.** Ohio CGM is irregular Dexcom samples; it is linearly
  interpolated onto a 5 min grid with gaps > 30 min NaN-bridged. Shanghai is
  similarly resampled to a 15 min grid with > 60 min gaps as NaN. AZT1D is
  natively 5-min and uses the same 30-min gap rule as Ohio. The simulator is
  already on a 5 min grid. All statistics ignore NaN.
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
- **AZT1D event log.** The pump CSV columns are parsed straight from `Subject
  N.csv`: Basal (U/hr), TotalBolusInsulinDelivered, CorrectionDelivered,
  FoodDelivered, CarbSize, BolusType, DeviceMode. Bolus events are flagged
  by a non-null BolusType. Meal vs correction-only is decided per event by
  `FoodDelivered > 0`. Device-mode time share counts every 5-min row.

---

## 3. Headline numbers

### 3.1 Pooled central moments

| Metric (mg/dL) | OhioT1DM | ShanghaiT1DM | AZT1D | T1DMSIM | Sim − Ohio | Sim − Shang | Sim − AZT1D |
|---|---:|---:|---:|---:|---:|---:|---:|
| n (samples) | 85,295 | 15,696 | 288,085 | 604,800 | — | — | — |
| **mean** | 162.1 | 164.7 | 148.2 | 160.4 | -1.7 | -4.3 | +12.3 |
| **median** | 155.2 | 156.6 | 139.2 | 148.9 | -6.3 | -7.7 | +9.8 |
| std | 60.8 | 72.3 | 47.5 | 76.0 | +15.1 | +3.6 | +28.4 |
| IQR | 86.2 | 106.2 | 58.0 | 99.3 | +13.1 | -6.9 | +41.3 |
| CV (%) | 37.5 | 43.9 | 32.1 | 47.4 | +9.8 pp | +3.5 pp | +15.3 pp |
| skewness | 0.58 | 0.51 | 1.03 | 1.07 | +0.49 | +0.56 | +0.04 |
| excess kurtosis | 0.15 | -0.14 | 1.54 | 1.44 | +1.29 | +1.58 | -0.09 |
| min | 40.0 | 39.6 | 40.0 | 22.1 | — | — | — |
| max | 400.0 | 475.2 | 400.0 | 500.0 | — | — | — |

### 3.2 Percentiles of the pooled distribution

| Percentile | OhioT1DM | ShanghaiT1DM | AZT1D | T1DMSIM | Sim − Ohio | Sim − Shang | Sim − AZT1D |
|---|---:|---:|---:|---:|---:|---:|---:|
| p1 | 57.0 | 41.3 | 67.1 | 55.2 | -1.8 | +13.9 | -11.9 |
| p5 | 76.0 | 61.2 | 86.0 | 65.3 | -10.7 | +4.1 | -20.7 |
| p10 | 88.0 | 75.6 | 97.0 | 74.9 | -13.1 | -0.7 | -22.1 |
| p25 | 115.4 | 108.0 | 115.0 | 101.4 | -14.0 | -6.6 | -13.7 |
| p50 | 155.2 | 156.6 | 139.2 | 148.9 | -6.3 | -7.7 | +9.8 |
| p75 | 201.6 | 214.2 | 173.0 | 200.7 | -0.9 | -13.5 | +27.7 |
| p90 | 244.6 | 264.6 | 211.8 | 262.9 | +18.3 | -1.7 | +51.1 |
| p95 | 271.0 | 291.6 | 240.0 | 305.4 | +34.4 | +13.8 | +65.4 |
| p99 | 325.8 | 349.2 | 294.8 | 395.3 | +69.5 | +46.1 | +100.5 |

![Percentile curves](figures/percentile_curves.png)

![Pooled PDF](figures/pdf_pooled.png)

![Pooled empirical CDF](figures/cdf_pooled.png)

![Q-Q vs each real cohort](figures/qq.png)

### 3.3 Distribution-distance statistics

| Pair | KS statistic | KS p-value | Wasserstein-1 (mg/dL) | JS divergence (5 mg/dL bins) |
|---|---:|---:|---:|---:|
| Ohio vs Shanghai | 0.063 | 3.5 × 10⁻⁴⁶ | 10.1 | 0.013 |
| Ohio vs AZT1D | 0.149 | < 10⁻³⁰⁰ | 17.3 | 0.025 |
| Shanghai vs AZT1D | 0.182 | < 10⁻³⁰⁰ | 26.3 | 0.056 |
| Sim vs Ohio | 0.082 | < 10⁻³⁰⁰ | 11.9 | 0.014 |
| Sim vs Shanghai | 0.061 | 3.2 × 10⁻⁴⁹ | 9.1 | 0.010 |
| Sim vs AZT1D | 0.127 | < 10⁻³⁰⁰ | 23.7 | 0.049 |

KS p-values fall to numerical zero in the right tail at these sample sizes
(Ohio ~85k, AZT1D ~320k, Sim ~600k); the magnitudes of the KS statistic and
the Wasserstein-1 distance are the meaningful quantities, not p.

---

## 4. Clinical glycemic indices

Per-record means ± std across each cohort.

| Index | OhioT1DM | ShanghaiT1DM | AZT1D | T1DMSIM |
|---|---|---|---|---|
| GMI / eA1c proxy | 7.19 ± 0.39 | 7.22 ± 0.74 | 6.87 ± 0.34 | 7.15 ± 0.57 |
| **LBGI** (low-BG risk) | 0.86 ± 0.49 | 1.82 ± 1.76 | 0.44 ± 0.28 | **1.54 ± 0.62** |
| **HBGI** (high-BG risk) | 7.60 ± 2.53 | 8.58 ± 4.28 | 4.81 ± 2.26 | **8.35 ± 3.85** |
| J-index | 49.2 ± 9.2 | 52.6 ± 17.1 | 37.9 ± 9.1 | 54.6 ± 16.3 |
| M-value (ref 120) | 11.1 ± 3.8 | 15.8 ± 6.2 | 5.8 ± 3.5 | 15.5 ± 7.2 |

Pooled (not per-record) risk indices, for reference:

| | Ohio | Shanghai | AZT1D | Sim |
|---|---:|---:|---:|---:|
| LBGI (pooled) | 0.85 | 1.87 | 0.45 | 1.54 |
| HBGI (pooled) | 7.54 | 8.87 | 4.74 | 8.35 |
| J-index (pooled) | 49.7 | 56.2 | 38.3 | 55.9 |
| M-value (pooled) | 11.0 | 16.5 | 5.7 | 15.5 |

### 4.1 Time-in-range, per-record cohort summary

| Range | OhioT1DM | ShanghaiT1DM | AZT1D | T1DMSIM |
|---|---|---|---|---|
| TBR2 (<54)        | 0.73 ± 0.68 | 2.79 ± 3.77 | 0.23 ± 0.32 | 0.73 ± 0.36 |
| TBR1 (54–70)      | 2.57 ± 1.61 | 4.72 ± 3.97 | 1.03 ± 0.90 | 6.63 ± 3.19 |
| **TIR (70–180)**  | **60.5 ± 10.2** | **54.7 ± 14.5** | **77.1 ± 10.5** | **59.1 ± 11.7** |
| TAR1 (180–250)    | 27.4 ± 6.1 | 25.1 ± 11.7 | 17.7 ± 6.8 | 21.3 ± 6.0 |
| TAR2 (>250)       | 8.88 ± 6.11 | 12.64 ± 8.91 | 3.95 ± 4.36 | 12.19 ± 8.26 |

The §0.3 class-balance stacked bar (`figures/class_balance.png`) plots the
same numbers visually for all four cohorts.

---

## 5. Variability and complexity

Per-record mean ± std.

| Metric (native cadence) | OhioT1DM | ShanghaiT1DM | AZT1D | T1DMSIM |
|---|---|---|---|---|
| CV (%)              | 36.2 ± 4.5   | 38.6 ± 6.8   | 29.9 ± 4.1   | **44.1 ± 6.8**   |
| MAGE (mg/dL)        | 103.9 ± 15.4     | 123.4 ± 30.0     | 80.6 ± 15.1     | 129.1 ± 20.4     |
| CONGA-1h (mg/dL)    | 39.4 ± 5.6 | 34.2 ± 7.2 | 37.6 ± 5.4 | 39.3 ± 4.7 |
| CONGA-4h (mg/dL)    | 76.1 ± 11.4 | 75.1 ± 17.7 | 63.4 ± 12.1 | 80.3 ± 9.8 |
| MODD (mg/dL)        | 61.1 ± 8.9     | 53.3 ± 12.8     | 42.6 ± 8.2     | **64.2 ± 11.9**     |
| Sample entropy      | 0.87 ± 0.10 | 0.44 ± 0.08¹ | 0.92 ± 0.12 | 0.80 ± 0.12 |

¹ Shanghai SampEn is computed on 15-min samples, which collapses the
  fine-scale jitter that drives SampEn at 5 min — the lower value is mostly a
  cadence artefact, not a real complexity difference.

![Variability and complexity panel](figures/variability_metrics.png)

---

## 6. Temporal dynamics

### 6.1 Autocorrelation

Pooled (mean across records) Pearson autocorrelation at the indicated lag.

| Lag         | OhioT1DM | ShanghaiT1DM | AZT1D | T1DMSIM |
|---|---|---|---|---|
| 5 min   | 0.995 | (n/a)  | 0.991 | 0.997 |
| 15 min  | 0.969 | 0.984  | 0.948 | 0.982 |
| 30 min  | 0.911 | 0.946  | 0.853 | 0.943 |
| 1 h     | 0.765 | 0.840  | 0.629 | 0.833 |
| 2 h     | 0.484 | 0.606  | 0.257 | 0.599 |
| 4 h     | 0.137 | 0.254  | -0.017 | 0.326 |
| **8 h**     | **-0.004** | **-0.028** | **-0.027** | **0.144** |
| **12 h**    | **-0.010** | **-0.050** | **-0.023** | **0.072** |
| 24 h    | 0.116 | 0.378  | 0.208 | **0.276** |

![Autocorrelation across lag](figures/acf.png)

### 6.2 Rate-of-change (Δ-BG)

![Δ-BG distribution at native cadence](figures/delta_distribution.png)

Per-record Δ-BG standard deviation (mean across records, native cadence):
Ohio 5.55 mg/dL · Shanghai 10.65 mg/dL ·
AZT1D 5.64 mg/dL · Sim 5.61 mg/dL.
Shanghai's value is at 15-min cadence and is not directly comparable to the
5-min values from Ohio, AZT1D, and the simulator.

### 6.3 Diurnal pattern (hour-of-day across records)

![Hour-of-day mean with ±1σ envelope](figures/diurnal_envelope.png)

![Hour-of-day median with IQR envelope](figures/diurnal_envelope_median.png)

Hour-by-hour mean BG (mg/dL):

|   | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Ohio | 149 | 151 | 155 | 160 | 164 | 169 | 173 | 179 | 186 | 178 | 164 | 153 | 154 | 161 | 166 | 165 | 163 | 160 | 162 | 162 | 157 | 158 | 156 | 151 |
| Shanghai | 166 | 164 | 163 | 159 | 156 | 158 | 165 | 169 | 192 | 175 | 144 | 137 | 149 | 143 | 147 | 157 | 166 | 179 | 184 | 175 | 170 | 169 | 168 | 167 |
| AZT1D | 147 | 142 | 139 | 135 | 131 | 130 | 129 | 134 | 146 | 157 | 159 | 150 | 142 | 151 | 164 | 164 | 157 | 148 | 152 | 161 | 163 | 159 | 157 | 154 |
| Sim | 130 | 133 | 140 | 151 | 163 | 173 | 178 | 178 | 176 | 171 | 167 | 165 | 167 | 169 | 169 | 165 | 162 | 163 | 167 | 168 | 163 | 154 | 144 | 134 |

Hour-by-hour median BG (mg/dL):

|   | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Ohio | 144 | 137 | 139 | 139 | 142 | 157 | 160 | 166 | 180 | 164 | 147 | 148 | 152 | 162 | 163 | 160 | 159 | 155 | 156 | 160 | 154 | 149 | 143 | 148 |
| Shanghai | 156 | 159 | 152 | 144 | 145 | 144 | 150 | 161 | 192 | 164 | 126 | 138 | 149 | 135 | 147 | 165 | 158 | 176 | 199 | 170 | 170 | 175 | 171 | 166 |
| AZT1D | 141 | 134 | 129 | 126 | 125 | 122 | 127 | 130 | 137 | 150 | 153 | 143 | 133 | 141 | 158 | 160 | 146 | 135 | 143 | 161 | 159 | 153 | 155 | 146 |
| Sim | 108 | 105 | 106 | 118 | 129 | 139 | 151 | 164 | 167 | 162 | 159 | 151 | 149 | 152 | 150 | 146 | 150 | 152 | 155 | 157 | 152 | 146 | 136 | 122 |

![Weekday × hour mean heatmap](figures/weekday_heatmap.png)

![Weekday × hour median heatmap](figures/weekday_heatmap_median.png)

---

## 7. Excursion-level dynamics

### 7.1 Episode counts and durations

Per-record means ± std.

| Metric | OhioT1DM | ShanghaiT1DM | AZT1D | T1DMSIM |
|---|---|---|---|---|
| Hypo (<70) episodes / day      | 0.81 ± 0.40 | 1.02 ± 0.74 | 0.46 ± 0.31 | **1.80 ± 0.73** |
| Severe-hypo (<54) eps / day   | 0.20 ± 0.20 | 0.51 ± 0.47 | 0.09 ± 0.10 | 0.28 ± 0.16 |
| Hyper (>180) episodes / day   | 2.61 ± 0.26 | 1.87 ± 0.71 | 2.68 ± 0.76 | 2.08 ± 0.43 |
| Severe-hyper (>250) eps / day | 1.06 ± 0.38 | 1.12 ± 0.68 | 0.62 ± 0.58 | 1.09 ± 0.49 |
| Hypo median duration (min)    | 33.3 | 69.4 | 25.9 | 45.9 |
| Hypo p90 duration (min)       | 89.8 | 179.2 | 47.8 | **94.2** |
| Hyper median duration (min)   | 131.2 | 213.3 | 79.3 | 139.8 |
| Hyper p90 duration (min)      | 422.8 | 622.6 | 237.0 | **511.0** |

![Episode duration boxplots](figures/episode_durations.png)

### 7.2 Hypo recovery time

![Hypo recovery time from BG<70 to BG≥80](figures/recovery_time.png)

Time from the first sub-70 sample to the next ≥ 80 sample:

| Cohort | n events | median (min) | p75 (min) | p90 (min) | p99 (min) | max (min) |
|---|---:|---:|---:|---:|---:|---:|
| Ohio     |   284 | 50 | 81 | 134 | 216 | 295 |
| Shanghai |   157 | 90 | 195 | 300 | 510 | 555 |
| AZT1D    |   617 | 30 | 50 | 75 | 177 | 235 |
| Sim      | 3,297 | 65 | 115 | 175 | 380 | 545 |

---

## 8. Per-record (per-patient) heterogeneity

![Per-record TIR vs TBR scatter](figures/per_patient_scatter.png)

| Cohort | TIR IQR (pp) | TIR min–max (pp) | Mean-BG std across records |
|---|---:|---|---:|
| Ohio     | 9.3 | 40.3 – 71.7 | 16.2 |
| Shanghai | 23.1 | 32.1 – 77.3 | 31.0 |
| AZT1D    | 14.9 | 44.6 – 92.6 | 14.1 |
| Sim      | 17.9 | 26.1 – 74.0 | 23.7 |

![LBGI and HBGI per-record boxplots](figures/risk_indices.png)

---

## 9. AZT1D insulin / carb behaviour panel

OhioT1DM and ShanghaiT1DM only expose CGM-level traces. AZT1D additionally
publishes the full pump event log — basal rate, bolus type (standard /
correction / automatic), units delivered for food and for correction, carb
size, and a device-mode flag (regular / sleep / exercise). The simulator
generates the same channels by construction (`basal_insulin`, `bolus_insulin`,
`total_carb`), so this section compares the comparable quantities head-to-head
and reports the AZT1D-only quantities alongside.

![AZT1D vs Sim insulin / carb panel](figures/azt1d_event_panel.png)

**Comparable quantities** (integrated daily totals + basal-rate distribution):

| Quantity | AZT1D | T1DMSIM | Δ (Sim − AZT1D) |
|---|---:|---:|---:|
| Mean basal rate (U/hr)            | 0.92 | 0.94 | +0.01 |
| Median basal rate (U/hr)          | 0.67 | 0.82 | +0.15 |
| Basal P10–P90 spread (U/hr)       | 0.20 – 2.03 | 0.36 – 1.69 | — |
| Carbs / day (g, per-subject mean) | 121.8 | 202.6 | +80.8 |
| Total insulin / day (U)           | 51.9 | 37.4 | -14.5 |

AZT1D's pooled basal-rate distribution was clipped at
10 U/hr before pooling
(3.2% of the basal column was discarded as
clearly non-physiological — values up to 4000+ U/hr in a few subjects,
likely PDF-to-CSV extraction artefacts). The Tandem AID adjusts basal every
5 min, so AZT1D's basal column is a long sequence of distinct rates; the
simulator's `basal_insulin` channel is the per-step PK-curve sample from
user-injected long-acting basal and integrates to the same daily total
**by construction** (matched to the patient's HGO via the
`basal = HGO_base × 24h × (BW/BW₀) × is_base / ICR` invariant), so this
panel is the sanity check that the daily-balance derivation lands in the
right ballpark for real T1D patients.

**AZT1D-only quantities** (user-initiated bolus events; AID auto-boluses are
filtered out — the simulator models MDI without a closed-loop controller and
has no auto-bolus channel to compare to):

| Quantity | AZT1D (per-subject mean) |
|---|---:|
| User-initiated boluses / day        | 7.22 |
| Meal boluses / day                  | 4.21 |
| Correction-only boluses / day       | 1.30 |
| Mean carbs / meal (g)               | 29.1 |
| Correction-unit share of total bolus | 13.8% |

Per-bolus and per-meal event counts are deliberately not computed for the
simulator: its `bolus_insulin` / `total_carb` channels expose per-step active
PK / absorption levels rather than discrete injection events, and any
threshold-based clustering merges events whose curves overlap. Comparison at
the **daily-total** level (above) is well-posed; comparison at the
per-event level is not.

- **Bolus type counts** (across the whole AZT1D pool, including AID-driven
  events):
  - Automatic Bolus/Correction: 3,478
  - Standard: 3,301
  - Standard/Correction: 2,746
  - BLE Standard Bolus/Correction: 787
  - BLE Standard Bolus: 749
  - Quick: 14
  - Extended 50.00%/0.00: 4
  - Extended 50.00%/12.00: 3
  - Extended/Correction 65.00%/3.18: 3
  - Extended 50.00%/20.00: 2
- **Device-mode time share:** regular 79.7%, sleep 19.7%, exercise 0.6%

---

## 10. Side-by-side summary

Raw deltas only — no qualitative verdicts. See sections 3–9 for context.

| Quantity | T1DMSIM | OhioT1DM | ShanghaiT1DM | AZT1D | Sim − Ohio | Sim − Shang | Sim − AZT1D |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pooled mean BG (mg/dL) | 160.4 | 162.1 | 164.7 | 148.2 | -1.7 | -4.3 | +12.3 |
| Pooled median BG (mg/dL) | 148.9 | 155.2 | 156.6 | 139.2 | -6.3 | -7.7 | +9.8 |
| Pooled std (mg/dL) | 76.0 | 60.8 | 72.3 | 47.5 | +15.1 | +3.6 | +28.4 |
| Pooled CV (%) | 47.4 | 37.5 | 43.9 | 32.1 | +9.8 | +3.5 | +15.3 |
| Pooled skewness | 1.07 | 0.58 | 0.51 | 1.03 | +0.49 | +0.56 | +0.04 |
| Pooled excess kurtosis | 1.44 | 0.15 | -0.14 | 1.54 | +1.29 | +1.58 | -0.09 |
| Pooled p99 (mg/dL) | 395.3 | 325.8 | 349.2 | 294.8 | +69.5 | +46.1 | +100.5 |
| GMI (per-record mean) | 7.15 | 7.19 | 7.22 | 6.87 | -0.05 | -0.08 | +0.28 |
| LBGI (per-record mean) | 1.54 | 0.86 | 1.82 | 0.44 | +0.68 | -0.29 | +1.10 |
| HBGI (per-record mean) | 8.35 | 7.60 | 8.58 | 4.81 | +0.75 | -0.23 | +3.53 |
| TIR % (per-record mean) | 59.1 | 60.5 | 54.7 | 77.1 | -1.3 | +4.4 | -17.9 |
| TBR1 % (per-record mean) | 6.63 | 2.57 | 4.72 | 1.03 | +4.06 | +1.91 | +5.60 |
| TBR2 % (per-record mean) | 0.73 | 0.73 | 2.79 | 0.23 | +0.00 | -2.05 | +0.50 |
| TAR1 % (per-record mean) | 21.3 | 27.4 | 25.1 | 17.7 | -6.1 | -3.8 | +3.6 |
| TAR2 % (per-record mean) | 12.2 | 8.9 | 12.6 | 3.9 | +3.3 | -0.5 | +8.2 |
| MAGE (mg/dL) | 129.1 | 103.9 | 123.4 | 80.6 | +25.3 | +5.8 | +48.6 |
| CONGA-1h (mg/dL) | 39.3 | 39.4 | 34.2 | 37.6 | -0.1 | +5.2 | +1.7 |
| CONGA-4h (mg/dL) | 80.3 | 76.1 | 75.1 | 63.4 | +4.2 | +5.3 | +16.9 |
| MODD (mg/dL) | 64.2 | 61.1 | 53.3 | 42.6 | +3.1 | +10.9 | +21.6 |
| Hypo episodes / day | 1.80 | 0.81 | 1.02 | 0.46 | +0.99 | +0.78 | +1.34 |
| Severe-hypo eps / day | 0.28 | 0.20 | 0.51 | 0.09 | +0.09 | -0.23 | +0.19 |
| Hyper episodes / day | 2.08 | 2.61 | 1.87 | 2.68 | -0.53 | +0.21 | -0.61 |
| Severe-hyper eps / day | 1.09 | 1.06 | 1.12 | 0.62 | +0.02 | -0.03 | +0.46 |
| Hypo p90 duration (min) | 94.2 | 89.8 | 179.2 | 47.8 | +4.5 | -85.0 | +46.5 |
| Hyper p90 duration (min) | 511.0 | 422.8 | 622.6 | 237.0 | +88.2 | -111.6 | +274.0 |
| Hypo recovery median (min) | 65.0 | 50.0 | 90.0 | 30.0 | +15.0 | -25.0 | +35.0 |
| Wasserstein-1 vs Ohio (mg/dL) | 11.9 | 10.1 | 9.1 | 17.3 | +1.8 | +2.8 | -5.4 |
| KS statistic vs Ohio | 0.082 | 0.063 | 0.061 | 0.149 | +0.019 | +0.021 | -0.067 |

---

## 11. Limitations of this comparison

- **Cohort size.** OhioT1DM (n = 6), ShanghaiT1DM (n = 16), and
  AZT1D (n = 24) are small enough that cohort means have non-trivial
  sampling error; each "real" distribution should be taken as a band, not a
  point. The simulator was exercised with 30 seeds for this report, but the
  generator is unbounded — production training pipelines can sample as many
  more seeds as they need.
- **Cadence asymmetry.** Shanghai's 15-min cadence collapses Δ-BG std and
  sample entropy relative to 5-min cohorts. Cross-cadence ACF below 30 min is
  not directly comparable.
- **No glucose-controller benchmark.** The simulator output is compared to
  three real human cohorts but not to UVA/Padova `simglucose` here.
- **AID asymmetry.** AZT1D subjects are all on closed-loop AID (Tandem
  Control-IQ); OhioT1DM is a mix of pump + announced meals; ShanghaiT1DM
  mixes CSII and MDI; the simulator models MDI long-acting basal + per-meal
  bolus. Differences in basal-rate variability and time-in-range partly
  reflect these different therapy regimens, not just simulator vs reality.
- **Sample entropy subsampling.** Records longer than 2,500 points are
  subsampled with `np.random.default_rng(0)` so the metric is reproducible but
  is a Monte-Carlo estimate, not the exact value over the full trace.

---

## 12. Reproduction

```bash
# regenerates diff/stats.json + diff/README.md + diff/figures/*.png
python diff/build_report.py
```

`scripts/compare_all_datasets.py` is reused for the dataset loaders and grid
regularisation. The three real datasets must live under `datasets/`
(`datasets/ohiot1dm/`, `datasets/ShanghaiT1DM/Shanghai_T1DM/`, and
`datasets/AZT1D/CGM Records/Subject N/`) — all gitignored, all subject to
their respective data-use agreements.

Numbers in this file come from one run of `build_report.py`
(30 seeds, 70 days each, 24 h warm-up discarded). Re-running reproduces
them exactly because the simulator is seed-deterministic and the real-data
side is fixed. Generating an arbitrarily larger synthetic corpus is just a
matter of bumping `n_seeds=` and/or `days=` in the `assemble_sim()` call.
