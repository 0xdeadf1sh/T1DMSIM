# T1DMSIM vs OhioT1DM, ShanghaiT1DM, AZT1D — Statistical Comparison Report

## 0. Machine-learning summary

### 0.1 Data volume per cohort

| Cohort | Records | Samples | Hours | CGM-days | Cadence |
|---|---:|---:|---:|---:|---:|
| OhioT1DM     |   6 | 85,295 |   7,108 |     296 | 5 min |
| ShanghaiT1DM |  16 | 15,696 |   3,924 |     164 | 15 min |
| AZT1D        |  25 | 300,884 |  25,074 |   1,045 | 5 min |
| **T1DMSIM** *(this run)* | **300** | **6,048,000** | **504,000** | **21,000** | **5 min** |

### 0.2 Normalization statistics

| Stat (mg/dL) | Ohio | Shanghai | AZT1D | **Sim** |
|---|---:|---:|---:|---:|
| mean    | 162.1  | 164.7  | 146.4  | **159.3**  |
| median  | 155.0  | 156.6  | 138.0  | **152.7**  |
| std     | 60.9   | 72.3   | 47.6   | **67.1**   |
| IQR     | 87.0   | 106.2   | 57.0   | **91.7**   |
| p1      | 57.0    | 41.3    | 65.0    | **48.0**    |
| p99     | 326.0   | 349.2   | 293.0   | **354.4**   |
| min     | 40.0   | 39.6   | 40.0   | **10.0**   |
| max     | 400.0   | 475.2   | 400.0   | **400.0**   |

### 0.3 Sample-level class balance

![Class balance per cohort](figures/class_balance.png)

| Band | Threshold | Ohio % | Shang % | AZT1D % | **Sim %** |
|---|---|---:|---:|---:|---:|
| TBR2 | <54     |  0.72 |  2.79 |  0.26 | **1.75** |
| TBR1 | 54-70   |  2.51 |  4.72 |  1.21 | **5.10** |
| TIR  | 70-180  |  60.7  |  54.7  |  77.8  | **58.4**  |
| TAR1 | 180-250 |  27.3 |  25.1 |  16.9 | **25.2** |
| TAR2 | >250    |   8.8 |  12.6 |   3.8 | **9.6** |

### 0.4 Episode-level event counts

| Event class | Ohio | Shanghai | AZT1D | **Sim** |
|---|---:|---:|---:|---:|
| Hypo (<70) episodes        | 252   | 169   | 570   | **25,182** |
| Severe-hypo (<54) episodes | 58    | 78    | 117    | **12,129** |
| Hyper (>180) episodes      | 847  | 305  | 2,752  | **46,386** |
| Severe-hyper (>250) episodes | 340  | 192  | 626  | **21,642** |

### 0.5 Effective context window

| ACF threshold | Ohio | Shanghai | AZT1D | **Sim** |
|---|---:|---:|---:|---:|
| 0.5 (50% retained) | 1.9 h | 2.6 h | 1.3 h | **2.4 h** |
| 0.2 (20% retained) | 3.6 h | 4.8 h | 2.4 h | **5.0 h** |

### 0.6 Cross-record heterogeneity

| Cohort | Between-patient mean-BG std | Within-patient BG std | Ratio |
|---|---:|---:|---:|
| Ohio     | 16.2 | 58.6 | 0.28 |
| Shanghai | 31.0 | 62.2 | 0.50 |
| AZT1D    | 16.0 | 44.1 | 0.36 |
| **Sim**  | **21.8** | **62.3** | **0.35** |

### 0.7 Diurnal shape (clean line overlay)

![Diurnal BG curves — clean line overlay](figures/diurnal_lines.png)

### 0.8 Sim-vs-real domain gap

| Pair | KS | Wasserstein-1 (mg/dL) | JS divergence |
|---|---:|---:|---:|
| **Sim vs Ohio**     | 0.050 | 5.5 | 0.008 |
| **Sim vs Shanghai** | 0.066 | 6.8 | 0.012 |
| **Sim vs AZT1D**    | 0.148 | 20.3 | 0.039 |
| Ohio vs Shanghai (real-vs-real baseline) | 0.064 | 10.1 | 0.018 |
| Ohio vs AZT1D    (real-vs-real baseline) | 0.157 | 18.1 | 0.025 |
| Shanghai vs AZT1D (real-vs-real baseline) | 0.190 | 26.9 | 0.060 |


## 1. Corpora at a glance

| Dataset | Records | Cadence | Total CGM-days | Cohort | Notes |
|---|---:|---:|---:|---|---|
| OhioT1DM | 6 records (file pairs) | 5 min Dexcom | 296.2 | US adults, pump + announced meals | training + testing periods concatenated per patient |
| ShanghaiT1DM | 16 records | **15 min** | 163.5 | CN adults, mixed CSII + MDI (incl. regular Novolin R), BMI ≈ 21 | shorter individual records (~10 d) |
| AZT1D | 25 subjects | 5 min Dexcom G6 | 1044.7 | US adults, Mayo Clinic AZ, all on Tandem t:slim X2 Control-IQ (AID) | rich pump event log: bolus type, basal rate, carbs, device mode |
| T1DMSIM *(this run)* | 300 seeds × 70 days | 5 min | 21000.0 | synthetic, seeds 0–299, 24 h warm-up discarded | `initial_bg = 120 mg/dL`, `bg_observed` (sensor-noised); generator is unbounded |

## 2. Headline numbers

### 2.1 Pooled central moments

| Metric (mg/dL) | OhioT1DM | ShanghaiT1DM | AZT1D | T1DMSIM | Sim − Ohio | Sim − Shang | Sim − AZT1D |
|---|---:|---:|---:|---:|---:|---:|---:|
| n (samples) | 85,295 | 15,696 | 300,884 | 6,048,000 | — | — | — |
| **mean** | 162.1 | 164.7 | 146.4 | 159.3 | -2.8 | -5.4 | +12.9 |
| **median** | 155.0 | 156.6 | 138.0 | 152.7 | -2.3 | -3.9 | +14.7 |
| std | 60.9 | 72.3 | 47.6 | 67.1 | +6.2 | -5.2 | +19.5 |
| IQR | 87.0 | 106.2 | 57.0 | 91.7 | +4.7 | -14.5 | +34.7 |
| CV (%) | 37.6 | 43.9 | 32.5 | 42.1 | +4.6 pp | -1.8 pp | +9.6 pp |
| skewness | 0.58 | 0.51 | 1.03 | 0.67 | +0.09 | +0.16 | -0.36 |
| excess kurtosis | 0.15 | -0.14 | 1.56 | 0.38 | +0.23 | +0.52 | -1.18 |
| min | 40.0 | 39.6 | 40.0 | 10.0 | — | — | — |
| max | 400.0 | 475.2 | 400.0 | 400.0 | — | — | — |

### 2.2 Percentiles of the pooled distribution

| Percentile | OhioT1DM | ShanghaiT1DM | AZT1D | T1DMSIM | Sim − Ohio | Sim − Shang | Sim − AZT1D |
|---|---:|---:|---:|---:|---:|---:|---:|
| p1 | 57.0 | 41.3 | 65.0 | 48.0 | -9.0 | +6.7 | -17.0 |
| p5 | 76.0 | 61.2 | 84.0 | 64.9 | -11.1 | +3.7 | -19.1 |
| p10 | 88.0 | 75.6 | 95.0 | 77.8 | -10.2 | +2.2 | -17.2 |
| p25 | 115.0 | 108.0 | 114.0 | 108.3 | -6.7 | +0.3 | -5.7 |
| p50 | 155.0 | 156.6 | 138.0 | 152.7 | -2.3 | -3.9 | +14.7 |
| p75 | 202.0 | 214.2 | 171.0 | 200.0 | -2.0 | -14.2 | +29.0 |
| p90 | 245.0 | 264.6 | 210.0 | 247.9 | +2.9 | -16.7 | +37.9 |
| p95 | 271.0 | 291.6 | 238.0 | 281.7 | +10.7 | -9.9 | +43.7 |
| p99 | 326.0 | 349.2 | 293.0 | 354.4 | +28.4 | +5.2 | +61.4 |

![Percentile curves](figures/percentile_curves.png)

![Pooled PDF](figures/pdf_pooled.png)

![Pooled empirical CDF](figures/cdf_pooled.png)

![Q-Q vs each real cohort](figures/qq.png)

### 2.3 Distribution-distance statistics

| Pair | KS statistic | KS p-value | Wasserstein-1 (mg/dL) | JS divergence (5 mg/dL bins) |
|---|---:|---:|---:|---:|
| Ohio vs Shanghai | 0.064 | 2.0 × 10⁻⁴⁷ | 10.1 | 0.018 |
| Ohio vs AZT1D | 0.157 | < 10⁻³⁰⁰ | 18.1 | 0.025 |
| Shanghai vs AZT1D | 0.190 | < 10⁻³⁰⁰ | 26.9 | 0.060 |
| Sim vs Ohio | 0.050 | 2.8 × 10⁻¹⁸¹ | 5.5 | 0.008 |
| Sim vs Shanghai | 0.066 | 7.9 × 10⁻⁶⁰ | 6.8 | 0.012 |
| Sim vs AZT1D | 0.148 | < 10⁻³⁰⁰ | 20.3 | 0.039 |

## 3. Clinical glycemic indices

| Index | OhioT1DM | ShanghaiT1DM | AZT1D | T1DMSIM |
|---|---|---|---|---|
| GMI / eA1c proxy | 7.19 ± 0.39 | 7.22 ± 0.74 | 6.83 ± 0.38 | 7.12 ± 0.52 |
| **LBGI** (low-BG risk) | 0.86 ± 0.49 | 1.82 ± 1.76 | 0.50 ± 0.39 | **1.68 ± 1.32** |
| **HBGI** (high-BG risk) | 7.60 ± 2.53 | 8.58 ± 4.28 | 4.65 ± 2.36 | **7.67 ± 3.26** |
| J-index | 49.2 ± 9.2 | 52.6 ± 17.1 | 37.1 ± 9.7 | 50.0 ± 13.4 |
| M-value (ref 120) | 11.1 ± 3.8 | 15.8 ± 6.2 | 5.8 ± 3.4 | 14.2 ± 6.7 |

| | Ohio | Shanghai | AZT1D | Sim |
|---|---:|---:|---:|---:|
| LBGI (pooled) | 0.86 | 1.87 | 0.51 | 1.68 |
| HBGI (pooled) | 7.54 | 8.87 | 4.56 | 7.67 |
| J-index (pooled) | 49.7 | 56.2 | 37.6 | 51.3 |
| M-value (pooled) | 11.0 | 16.5 | 5.7 | 14.2 |

### 3.1 Time-in-range, per-record cohort summary

| Range | OhioT1DM | ShanghaiT1DM | AZT1D | T1DMSIM |
|---|---|---|---|---|
| TBR2 (<54)        | 0.72 ± 0.66 | 2.79 ± 3.77 | 0.26 ± 0.32 | 1.75 ± 1.92 |
| TBR1 (54–70)      | 2.51 ± 1.57 | 4.72 ± 3.97 | 1.21 ± 1.26 | 5.10 ± 4.40 |
| **TIR (70–180)**  | **60.7 ± 10.2** | **54.7 ± 14.5** | **77.8 ± 10.7** | **58.4 ± 11.1** |
| TAR1 (180–250)    | 27.3 ± 6.1 | 25.1 ± 11.7 | 16.9 ± 7.4 | 25.2 ± 7.8 |
| TAR2 (>250)       | 8.83 ± 6.09 | 12.64 ± 8.91 | 3.77 ± 4.32 | 9.60 ± 7.24 |

## 4. Variability and complexity

| Metric (native cadence) | OhioT1DM | ShanghaiT1DM | AZT1D | T1DMSIM |
|---|---|---|---|---|
| CV (%)              | 36.3 ± 4.5   | 38.6 ± 6.8   | 29.7 ± 4.2   | **39.4 ± 7.2**   |
| MAGE (mg/dL)        | 102.6 ± 14.9     | 123.4 ± 30.0¹     | 78.2 ± 16.1     | 111.8 ± 25.0     |
| CONGA-1h (mg/dL)    | 39.5 ± 5.6 | 34.2 ± 7.2 | 37.5 ± 5.5 | 36.0 ± 8.4 |
| CONGA-4h (mg/dL)    | 76.2 ± 11.5 | 75.1 ± 17.7 | 62.5 ± 12.9 | 76.4 ± 15.7 |
| MODD (mg/dL)        | 61.1 ± 8.9     | 53.3 ± 12.8     | 42.1 ± 8.5     | **61.4 ± 13.4**     |
| Sample entropy      | 0.56 ± 0.09 | 0.44 ± 0.08 | 0.75 ± 0.11 | 0.53 ± 0.07 |

![Variability and complexity panel](figures/variability_metrics.png)

## 5. Temporal dynamics

### 5.1 Autocorrelation

| Lag         | OhioT1DM | ShanghaiT1DM | AZT1D | T1DMSIM |
|---|---|---|---|---|
| 5 min   | 0.995 | (n/a)  | 0.990 | 0.996 |
| 15 min  | 0.968 | 0.984  | 0.943 | 0.982 |
| 30 min  | 0.909 | 0.946  | 0.846 | 0.943 |
| 1 h     | 0.764 | 0.840  | 0.617 | 0.828 |
| 2 h     | 0.483 | 0.606  | 0.248 | 0.567 |
| 4 h     | 0.137 | 0.254  | -0.015 | 0.237 |
| **8 h**     | **-0.004** | **-0.028** | **-0.021** | **0.080** |
| **12 h**    | **-0.010** | **-0.050** | **-0.020** | **0.030** |
| 24 h    | 0.116 | 0.378  | 0.203 | **0.200** |

![Autocorrelation across lag](figures/acf.png)

### 5.2 Rate-of-change (Δ-BG)

![Δ-BG distribution at native cadence](figures/delta_distribution.png)

### 5.3 Diurnal pattern (hour-of-day across records)

![Hour-of-day mean with ±1σ envelope](figures/diurnal_envelope.png)

![Hour-of-day median with IQR envelope](figures/diurnal_envelope_median.png)

|   | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Ohio | 149 | 151 | 155 | 160 | 164 | 169 | 173 | 179 | 186 | 178 | 164 | 153 | 154 | 161 | 166 | 165 | 163 | 160 | 162 | 162 | 157 | 158 | 156 | 152 |
| Shanghai | 166 | 164 | 163 | 159 | 156 | 158 | 165 | 169 | 192 | 175 | 144 | 137 | 149 | 143 | 147 | 157 | 166 | 179 | 184 | 175 | 170 | 169 | 168 | 167 |
| AZT1D | 145 | 141 | 137 | 134 | 130 | 128 | 128 | 132 | 144 | 155 | 157 | 148 | 141 | 149 | 161 | 162 | 155 | 147 | 150 | 158 | 161 | 157 | 155 | 152 |
| Sim | 140 | 139 | 138 | 140 | 147 | 159 | 174 | 186 | 194 | 195 | 190 | 182 | 173 | 166 | 160 | 156 | 153 | 152 | 151 | 149 | 148 | 147 | 145 | 142 |

|   | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Ohio | 144 | 137 | 139 | 139 | 142 | 157 | 160 | 166 | 180 | 165 | 147 | 148 | 152 | 162 | 163 | 160 | 159 | 155 | 156 | 160 | 154 | 149 | 143 | 148 |
| Shanghai | 156 | 159 | 152 | 144 | 145 | 144 | 150 | 161 | 192 | 164 | 126 | 138 | 149 | 135 | 147 | 165 | 158 | 176 | 199 | 170 | 170 | 175 | 171 | 166 |
| AZT1D | 140 | 131 | 129 | 126 | 125 | 121 | 126 | 129 | 137 | 150 | 151 | 141 | 133 | 140 | 156 | 160 | 146 | 134 | 142 | 159 | 154 | 152 | 155 | 144 |
| Sim | 132 | 130 | 125 | 127 | 135 | 149 | 168 | 184 | 192 | 193 | 188 | 179 | 168 | 159 | 152 | 148 | 143 | 141 | 139 | 140 | 140 | 138 | 138 | 136 |

![Weekday × hour mean heatmap](figures/weekday_heatmap.png)

![Weekday × hour median heatmap](figures/weekday_heatmap_median.png)

## 6. Excursion-level dynamics

### 6.1 Episode counts and durations

| Metric | OhioT1DM | ShanghaiT1DM | AZT1D | T1DMSIM |
|---|---|---|---|---|
| Hypo (<70) episodes / day      | 0.86 ± 0.43 | 1.02 ± 0.74 | 0.53 ± 0.45 | **1.20 ± 0.82** |
| Severe-hypo (<54) eps / day   | 0.20 ± 0.21 | 0.51 ± 0.47 | 0.11 ± 0.12 | 0.58 ± 0.64 |
| Hyper (>180) episodes / day   | 2.86 ± 0.27 | 1.87 ± 0.71 | 2.65 ± 0.89 | 2.21 ± 0.49 |
| Severe-hyper (>250) eps / day | 1.17 ± 0.43 | 1.12 ± 0.68 | 0.62 ± 0.59 | 1.03 ± 0.54 |
| Hypo median duration (min)    | 34.6 | 69.4 | 26.6 | 66.2 |
| Hypo p90 duration (min)       | 89.6 | 179.2 | 49.3 | **126.2** |
| Hyper median duration (min)   | 127.9 | 213.3 | 75.5 | 147.1 |
| Hyper p90 duration (min)      | 421.8 | 622.6 | 226.9 | **511.0** |

![Episode duration boxplots](figures/episode_durations.png)

### 6.2 Hypo recovery time

![Hypo recovery time from BG<70 to BG≥80](figures/recovery_time.png)

| Cohort | n events | median (min) | p75 (min) | p90 (min) | p99 (min) | max (min) |
|---|---:|---:|---:|---:|---:|---:|
| Ohio     |   218 | 55 | 90 | 145 | 210 | 290 |
| Shanghai |   156 | 90 | 195 | 300 | 510 | 555 |
| AZT1D    |   517 | 40 | 55 | 90 | 193 | 235 |
| Sim      | 21,047 | 90 | 140 | 210 | 455 | 1530 |

### 6.3 Unexplained excursions

![Unexplained-excursion summary](figures/unexplained_summary.png)

![Unexplained excursions in real CGM](figures/unexplained_gallery.png)

| Quantity | OhioT1DM | AZT1D | T1DMSIM |
|---|---:|---:|---:|
| Excursions detected | 2266 | 8770 | 113992 |
| &nbsp;&nbsp;rises unexplained (%) | 60.7 | 79.2 | 66.3 |
| &nbsp;&nbsp;falls unexplained (%) | 35.4 | 28.0 | 71.3 |
| &nbsp;&nbsp;all unexplained (%) | 48.9 | 54.4 | 68.8 |
| Explained load (mg/dL/day) | 393 | 364 | 209 |
| Unexplained load (mg/dL/day) | 334 | 400 | 436 |
| Median amplitude, explained (mg/dL) | 96 | 86 | 109 |
| Median amplitude, unexplained (mg/dL) | 83 | 78 | 101 |
| Δ-BG SD, full trace (mg/dL) | 5.89 | 6.02 | 5.23 |
| Δ-BG SD, unexplained censored (mg/dL) | 5.85 | 5.64 | 5.23 |

## 7. Per-record (per-patient) heterogeneity

![Per-record TIR vs TBR scatter](figures/per_patient_scatter.png)

| Cohort | TIR IQR (pp) | TIR min–max (pp) | Mean-BG std across records |
|---|---:|---|---:|
| Ohio     | 9.4 | 40.6 – 71.8 | 16.2 |
| Shanghai | 23.1 | 32.1 – 77.3 | 31.0 |
| AZT1D    | 14.9 | 44.7 – 92.7 | 16.0 |
| Sim      | 14.9 | 20.0 – 89.4 | 21.8 |

![LBGI and HBGI per-record boxplots](figures/risk_indices.png)

## 8. AZT1D insulin / carb behaviour panel

![AZT1D vs Sim insulin / carb panel](figures/azt1d_event_panel.png)

| Quantity | AZT1D | T1DMSIM | Δ (Sim − AZT1D) |
|---|---:|---:|---:|
| Mean basal rate (U/hr)            | 0.92 | 1.47 | +0.55 |
| Median basal rate (U/hr)          | 0.67 | 1.16 | +0.49 |
| Basal P10–P90 spread (U/hr)       | 0.20 – 2.03 | 0.31 – 3.08 | — |
| Carbs / day (g, per-subject mean) | 121.8 | 225.6 | +103.8 |
| Total insulin / day (U)           | 51.9 | 54.6 | +2.7 |

| Quantity | AZT1D (per-subject mean) |
|---|---:|
| User-initiated boluses / day        | 7.22 |
| Meal boluses / day                  | 4.21 |
| Correction-only boluses / day       | 1.30 |
| Mean carbs / meal (g)               | 29.1 |
| Correction-unit share of total bolus | 13.8% |

| Bolus type (whole AZT1D pool, incl. AID-driven) | Count |
|---|---:|
| Automatic Bolus/Correction | 3,478 |
| Standard | 3,301 |
| Standard/Correction | 2,746 |
| BLE Standard Bolus/Correction | 787 |
| BLE Standard Bolus | 749 |
| Quick | 14 |
| Extended 50.00%/0.00 | 4 |
| Extended 50.00%/12.00 | 3 |
| Extended/Correction 65.00%/3.18 | 3 |
| Extended 50.00%/20.00 | 2 |

| Device-mode time share | AZT1D % |
|---|---:|
| regular | 79.7 |
| sleep | 19.7 |
| exercise | 0.6 |

## 9. Side-by-side summary

| Quantity | T1DMSIM | OhioT1DM | ShanghaiT1DM | AZT1D | Sim − Ohio | Sim − Shang | Sim − AZT1D |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pooled mean BG (mg/dL) | 159.3 | 162.1 | 164.7 | 146.4 | -2.8 | -5.4 | +12.9 |
| Pooled median BG (mg/dL) | 152.7 | 155.0 | 156.6 | 138.0 | -2.3 | -3.9 | +14.7 |
| Pooled std (mg/dL) | 67.1 | 60.9 | 72.3 | 47.6 | +6.2 | -5.2 | +19.5 |
| Pooled CV (%) | 42.1 | 37.6 | 43.9 | 32.5 | +4.6 | -1.8 | +9.6 |
| Pooled skewness | 0.67 | 0.58 | 0.51 | 1.03 | +0.09 | +0.16 | -0.36 |
| Pooled excess kurtosis | 0.38 | 0.15 | -0.14 | 1.56 | +0.23 | +0.52 | -1.18 |
| Pooled p99 (mg/dL) | 354.4 | 326.0 | 349.2 | 293.0 | +28.4 | +5.2 | +61.4 |
| GMI (per-record mean) | 7.12 | 7.19 | 7.22 | 6.83 | -0.07 | -0.10 | +0.29 |
| LBGI (per-record mean) | 1.68 | 0.86 | 1.82 | 0.50 | +0.81 | -0.15 | +1.18 |
| HBGI (per-record mean) | 7.67 | 7.60 | 8.58 | 4.65 | +0.07 | -0.91 | +3.02 |
| TIR % (per-record mean) | 58.4 | 60.7 | 54.7 | 77.8 | -2.3 | +3.6 | -19.5 |
| TBR1 % (per-record mean) | 5.10 | 2.51 | 4.72 | 1.21 | +2.59 | +0.38 | +3.89 |
| TBR2 % (per-record mean) | 1.75 | 0.72 | 2.79 | 0.26 | +1.03 | -1.04 | +1.49 |
| TAR1 % (per-record mean) | 25.2 | 27.3 | 25.1 | 16.9 | -2.1 | +0.1 | +8.3 |
| TAR2 % (per-record mean) | 9.6 | 8.8 | 12.6 | 3.8 | +0.8 | -3.0 | +5.8 |
| MAGE (mg/dL) | 111.8 | 102.6 | 123.4 | 78.2 | +9.2 | -11.6 | +33.6 |
| CONGA-1h (mg/dL) | 36.0 | 39.5 | 34.2 | 37.5 | -3.5 | +1.8 | -1.5 |
| CONGA-4h (mg/dL) | 76.4 | 76.2 | 75.1 | 62.5 | +0.2 | +1.3 | +13.9 |
| MODD (mg/dL) | 61.4 | 61.1 | 53.3 | 42.1 | +0.2 | +8.1 | +19.3 |
| Hypo episodes / day | 1.20 | 0.86 | 1.02 | 0.53 | +0.34 | +0.18 | +0.67 |
| Severe-hypo eps / day | 0.58 | 0.20 | 0.51 | 0.11 | +0.38 | +0.07 | +0.47 |
| Hyper episodes / day | 2.21 | 2.86 | 1.87 | 2.65 | -0.65 | +0.34 | -0.44 |
| Severe-hyper eps / day | 1.03 | 1.17 | 1.12 | 0.62 | -0.13 | -0.09 | +0.41 |
| Hypo p90 duration (min) | 126.2 | 89.6 | 179.2 | 49.3 | +36.6 | -53.0 | +77.0 |
| Hyper p90 duration (min) | 511.0 | 421.8 | 622.6 | 226.9 | +89.2 | -111.6 | +284.1 |
| Hypo recovery median (min) | 90.0 | 55.0 | 90.0 | 40.0 | +35.0 | +0.0 | +50.0 |

| Pooled distance | Sim vs Ohio | Sim vs Shanghai | Sim vs AZT1D | Ohio–Shang | Ohio–AZT1D | Shang–AZT1D |
|---|---:|---:|---:|---:|---:|---:|
| Wasserstein-1 (mg/dL) | 5.5 | 6.8 | 20.3 | 10.1 | 18.1 | 26.9 |
| KS statistic | 0.050 | 0.066 | 0.148 | 0.064 | 0.157 | 0.190 |

## 10. Extended statistics

### 10.1 Cadence-fair variability (common 15-min grid)

| Metric (15-min grid) | OhioT1DM | ShanghaiT1DM | AZT1D | T1DMSIM |
|---|---:|---:|---:|---:|
| MAGE (mg/dL) | 109.5 | 123.4 | 87.3 | **118.8** |
| Δ-BG SD, 15-min (mg/dL) | 14.55 | 10.65 | 14.26 | **11.76** |
| CONGA-1h (mg/dL) | 39.55 | 34.17 | 37.45 | **35.97** |
| ACF @ 30 min | 0.909 | 0.946 | 0.846 | **0.943** |
| ACF @ 60 min | 0.764 | 0.840 | 0.617 | **0.828** |
| ACF @ 120 min | 0.483 | 0.606 | 0.248 | **0.567** |
| Hypo eps / day | 0.96 | 1.02 | 0.64 | **1.22** |
| Hyper eps / day | 2.81 | 1.87 | 2.74 | **2.22** |

### 10.2 Additional two-sample distances (pooled BG)

| Distance | Sim vs Ohio | Sim vs Shang | Sim vs AZT1D | Ohio–Shang | Ohio–AZT1D | Shang–AZT1D |
|---|---:|---:|---:|---:|---:|---:|
| Energy distance | 0.544 | 0.724 | 1.943 | 0.910 | 1.971 | 2.594 |
| Cramér–von Mises | 8.0 | 7.5 | 66.2 | 11.6 | 75.9 | 107.7 |
| Anderson–Darling | 76.1 | 58.2 | 544.1 | 133.3 | 525.0 | 875.1 |
| Total variation | 0.066 | 0.102 | 0.226 | 0.127 | 0.191 | 0.279 |
| Hellinger | 0.092 | 0.113 | 0.201 | 0.140 | 0.160 | 0.251 |
| Histogram overlap | 0.934 | 0.898 | 0.774 | 0.873 | 0.809 | 0.721 |

### 10.3 Temporal structure (common 15-min grid)

| Metric | OhioT1DM | ShanghaiT1DM | AZT1D | T1DMSIM |
|---|---:|---:|---:|---:|
| Poincaré SD1 (mg/dL) | 10.29 | 7.53 | 10.08 | **8.31** |
| Poincaré SD2 (mg/dL) | 82.2 | 87.7 | 61.5 | **87.7** |
| Poincaré SD1/SD2 | 0.126 | 0.088 | 0.169 | **0.096** |
| Spectral entropy (0–1) | 0.530 | 0.496 | 0.622 | **0.530** |
| Spectral centroid (cyc/h) | 0.110 | 0.085 | 0.150 | **0.091** |
| DFA α (Hurst) | 1.432 | 1.537 | 1.325 | **1.522** |
| ACF e-folding (1/e) | 2.7 h | 3.4 h | 1.7 h | **3.2 h** |

![Band transition heatmaps](figures/band_transitions.png)

| Band | OhioT1DM | ShanghaiT1DM | AZT1D | T1DMSIM |
|---|---:|---:|---:|---:|
| TBR2 | 40 | 85 | 26 | 37 |
| TBR1 | 32 | 49 | 26 | 43 |
| TIR | 249 | 271 | 339 | 245 |
| TAR1 | 106 | 122 | 72 | 111 |
| TAR2 | 108 | 165 | 84 | 129 |

| Transition-matrix distance from Sim (Frobenius) | OhioT1DM | ShanghaiT1DM | AZT1D |
|---|---:|---:|---:|
| Distance | 0.200 | 0.330 | 0.479 |

### 10.4 Cross-seed bootstrap 95% CIs

| Statistic | OhioT1DM | ShanghaiT1DM | AZT1D | T1DMSIM |
|---|---|---|---|---|
| Pooled mean (mg/dL) | 162.1 [151.0, 177.6] | 164.7 [149.7, 180.8] | 146.4 [140.1, 153.3] | **159.3 [156.9, 161.9]** |
| Pooled std (mg/dL) | 60.9 [52.9, 66.0] | 72.3 [66.0, 77.2] | 47.6 [43.3, 52.1] | **67.1 [65.5, 69.0]** |
| TIR % (70–180) | 60.9 [51.1, 68.3] | 53.6 [46.0, 60.8] | 78.2 [73.6, 82.2] | **58.4 [57.0, 59.5]** |
| LBGI | 0.86 [0.49, 1.29] | 1.87 [0.98, 2.74] | 0.51 [0.37, 0.67] | **1.68 [1.54, 1.83]** |

### 10.5 Standardised strength / weakness gap score

![Standardised gap score](figures/gap_score.png)

| Metric | Sim | Ohio | Shanghai | AZT1D | z | within envelope (±1% of span) |
|---|---:|---:|---:|---:|---:|:--:|
| cf_hypo_per_day | 1.22 | 0.96 | 1.02 | 0.64 | +1.67 | **no** |
| TBR1% | 5.10 | 2.49 | 4.82 | 1.25 | +1.24 | **no** |
| dfa_alpha | 1.52 | 1.43 | 1.54 | 1.33 | +0.85 | yes |
| LBGI | 1.68 | 0.86 | 1.87 | 0.51 | +0.85 | yes |
| sd_ratio | 0.10 | 0.13 | 0.09 | 0.17 | -0.78 | yes |
| acf_efold_min | 192.46 | 159.98 | 201.21 | 100.47 | +0.76 | yes |
| cv_pct | 42.12 | 37.55 | 43.89 | 32.51 | +0.72 | yes |
| cf_mage | 118.79 | 109.49 | 123.38 | 87.27 | +0.66 | yes |
| cf_delta_std | 11.76 | 14.55 | 10.65 | 14.26 | -0.64 | yes |
| std | 67.10 | 60.86 | 72.31 | 47.60 | +0.55 | yes |
| cf_hyper_per_day | 2.22 | 2.81 | 1.87 | 2.74 | -0.49 | yes |
| TIR% | 58.37 | 60.93 | 53.62 | 78.25 | -0.47 | yes |
| cf_conga_1h | 35.97 | 39.55 | 34.17 | 37.45 | -0.40 | yes |
| TBR2% | 1.75 | 0.71 | 2.83 | 0.27 | +0.35 | yes |
| HBGI | 7.67 | 7.54 | 8.87 | 4.56 | +0.31 | yes |
| spectral_entropy | 0.53 | 0.53 | 0.50 | 0.62 | -0.29 | yes |
| TAR2% | 9.60 | 8.63 | 13.48 | 3.65 | +0.20 | yes |
| mean | 159.30 | 162.07 | 164.75 | 146.42 | +0.16 | yes |
| excess_kurt | 0.38 | 0.15 | -0.14 | 1.56 | -0.16 | yes |
| skew | 0.67 | 0.58 | 0.51 | 1.03 | -0.14 | yes |

