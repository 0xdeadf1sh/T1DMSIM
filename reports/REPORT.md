# T1DMSIM vs OhioT1DM vs ShanghaiT1DM — Statistical Comparison Report

Comprehensive statistical comparison of the synthetic blood-glucose traces
produced by `simulator.py` against two non-redistributable real-world CGM
corpora. Goes well beyond the summary panel in `README.md`: it adds full
percentile tables, distribution-distance statistics (Kolmogorov–Smirnov,
Wasserstein, Jensen–Shannon), Kovatchev risk indices, MAGE / CONGA / MODD /
sample entropy, autocorrelation across nine lags, rate-of-change distributions,
hour-of-day envelopes, weekday × hour heatmaps, per-record TIR/TBR scatter, and
expanded excursion-level metrics.

All numbers in this report are produced by `reports/build_report.py`. Raw stats
are persisted to `reports/stats.json`; figures live in `reports/figures/`.

---

## 1. Corpora at a glance

| Dataset | Records | Cadence | Total CGM-days | Cohort | Notes |
|---|---:|---:|---:|---|---|
| OhioT1DM | 6 patients (12 XML files) | 5 min Dexcom | 321.7 | US adults, pump + announced meals | training + testing periods concatenated per patient |
| ShanghaiT1DM | 16 records / 13 patients | **15 min** | 163.5 | CN adults, mixed CSII + MDI (incl. regular Novolin R), BMI ≈ 21 | shorter individual records (~10 d) |
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
  are computed at each cohort’s **native** cadence. Cross-cadence comparison is
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
| **mean** | 162.1 | 164.7 | 174.4 | +12.4 | +9.7 |
| **median** | 155.2 | 156.6 | **150.6** | −4.6 | −6.0 |
| std | 60.8 | 72.3 | 89.7 | +28.9 | +17.4 |
| IQR | 86.2 | 106.2 | 109.9 | +23.7 | +3.7 |
| CV (%) | 37.5 | 43.9 | 51.4 | +13.9 pp | +7.5 pp |
| skewness | 0.58 | 0.51 | **1.18** | +0.60 | +0.67 |
| excess kurtosis | 0.15 | −0.14 | **1.20** | +1.05 | +1.34 |
| min | 40.0 | 39.6 | 20.0 | — | — |
| max | 400.0 | 475.2 | 500.0 | — | — |

The median sits within ~5 mg/dL of both real cohorts. The mean runs ~10
mg/dL above both real cohorts — the simulator's body of the distribution
matches real but its right tail extends further. Skewness (1.18) and
excess kurtosis (1.20) remain substantially above real cohorts (0.5 / 0.0):
the synthetic distribution is right-skewed and leptokurtic where the real
ones are nearly symmetric.

### 3.2 Percentiles of the pooled distribution

| Percentile | OhioT1DM | ShanghaiT1DM | T1DMSIM |
|---|---:|---:|---:|
| p1  | 57.0  | 41.3  | 55.7  |
| p5  | 76.0  | 61.2  | 68.7  |
| p10 | 88.0  | 75.6  | 81.1  |
| p25 | 115.4 | 108.0 | 109.5 |
| p50 | 155.2 | 156.6 | 150.6 |
| p75 | 201.6 | 214.2 | 219.4 |
| p90 | 244.6 | 264.6 | 304.1 |
| p95 | 271.0 | 291.6 | 355.4 |
| p99 | 325.8 | 349.2 | 464.6 |

Versus Ohio the simulator sits 5–8 mg/dL **below** through the lower body
(p1–p50), crosses over between p50 and p75, then runs 18 mg/dL above at p75
and 40–139 mg/dL above at p90–p99. Versus Shanghai the simulator sits slightly
**above** through the lower body (the Shanghai cohort has a longer left tail —
p5 = 61 mg/dL vs Sim p5 = 69), barely below at the median, then crosses over
at p75 and remains above out to the right tail. The crossover against either
real cohort is between p50 and p75.

![Percentile curves](figures/percentile_curves.png)

![Pooled PDF](figures/pdf_pooled.png)

![Pooled empirical CDF](figures/cdf_pooled.png)

![Q-Q vs Ohio and Shanghai](figures/qq.png)

The Q-Q plots make the shape mismatch unambiguous: the Sim quantile curve sits
*below* y=x in the body (sim is lower than reference at the same probability
mass) and *above* y=x in the right tail (sim is higher).

### 3.3 Distribution-distance statistics

| Pair | KS statistic | KS p-value | Wasserstein-1 (mg/dL) | JS divergence (5 mg/dL bins) |
|---|---:|---:|---:|---:|
| Ohio vs Shanghai | 0.063 | 3.5 × 10⁻⁴⁶ | 10.1 | 0.013 |
| Sim vs Ohio      | 0.098 | < 10⁻³⁰⁰ | 18.9 | 0.019 |
| Sim vs Shanghai  | 0.067 | 3.2 × 10⁻⁶⁰ | 13.1 | 0.015 |

Sim is roughly **1.5–2× farther** from either real cohort than the two real
cohorts are from each other on all three distance metrics. KS p-values are
essentially zero everywhere because of the sample sizes (Ohio 85k, Sim 600k)
— the *magnitude* of the KS statistic and the Wasserstein gap are the
meaningful numbers, not p.

---

## 4. Clinical glycemic indices

Per-record means ± std across each cohort.

| Index | OhioT1DM | ShanghaiT1DM | T1DMSIM |
|---|---|---|---|
| GMI / eA1c proxy | 7.19 ± 0.39 | 7.22 ± 0.74 | 7.48 ± 0.73 |
| **LBGI** (low-BG risk) | 0.86 ± 0.49 | 1.82 ± 1.76 | **1.18 ± 0.44** |
| **HBGI** (high-BG risk) | 7.60 ± 2.53 | 8.58 ± 4.28 | **10.91 ± 5.52** |
| J-index | 49.2 ± 9.2 | 52.6 ± 17.1 | 68.0 ± 25.0 |
| M-value (ref 120) | 11.1 ± 3.8 | 15.8 ± 6.2 | 21.6 ± 12.1 |

- **GMI sits 0.26 above both real cohorts**, consistent with the +10 mg/dL
  mean BG offset.
- **LBGI is between the two real cohorts** (1.18 vs Ohio 0.86 and Shanghai
  1.82). The simulator is meaningfully riskier on the hypo side than Ohio
  but safer than Shanghai.
- **HBGI is above both real cohorts** (10.9 vs Ohio 7.6 and Shanghai 8.6).
  HBGI grows as the square of the deviation, so the simulator's heavy
  right tail (p99 = 465 mg/dL) directly drives this gap.
- The simulator's **inter-record variance** of HBGI is roughly **2× higher
  than Ohio** (σ 5.5 vs 2.5). A small number of synthetic patients run
  multi-day stretches above 250 mg/dL while the rest of the cohort sits in
  clinical range.

### 4.1 Time-in-range, per-record cohort summary

| Range | OhioT1DM | ShanghaiT1DM | T1DMSIM |
|---|---|---|---|
| TBR2 (<54)        | 0.73 ± 0.68  | 2.79 ± 3.77  | 0.68 ± 0.38  |
| TBR1 (54–70)      | 2.57 ± 1.61  | 4.72 ± 3.97  | 4.82 ± 1.77  |
| **TIR (70–180)**  | **60.5 ± 10.2** | **54.7 ± 14.5** | **57.4 ± 13.5** |
| TAR1 (180–250)    | 27.4 ± 6.1   | 25.1 ± 11.7  | 18.8 ± 5.3   |
| TAR2 (>250)       | 8.88 ± 6.11  | 12.64 ± 8.91 | 18.28 ± 11.51|

The simulator's TIR sits within the real-cohort band (Ohio 60.5 / Shanghai
54.7). TBR2 closely matches Ohio and TBR1 matches Shanghai. The remaining
gaps are (a) under-represented TAR1 (19 % vs Ohio 27 % / Shang 25 %) and
(b) over-represented TAR2 (18 % vs Ohio 9 % / Shang 13 %).

![Clinical-range cohort comparison](../assets/clinical_ranges.png)

(The bar chart from the README is included here for direct reference; the
figure is produced by `scripts/generate_comparison_figures.py`.)

---

## 5. Variability and complexity

Per-record mean ± std.

| Metric (native cadence) | OhioT1DM | ShanghaiT1DM | T1DMSIM |
|---|---|---|---|
| CV (%)              | 36.2 ± 4.5   | 38.6 ± 6.8  | **47.0 ± 6.3**  |
| MAGE (mg/dL)        | 103.9 ± 15.4 | 123.4 ± 30.0| 147.9 ± 27.1 |
| CONGA-1h (mg/dL)    | 39.4 ± 5.6   | 34.2 ± 7.2  | 49.7 ± 6.1   |
| CONGA-4h (mg/dL)    | 76.1 ± 11.4  | 75.1 ± 17.7 | 93.6 ± 12.9  |
| MODD (mg/dL)        | 61.1 ± 8.9   | 53.3 ± 12.8 | **81.0 ± 19.1** |
| Sample entropy      | 0.87 ± 0.10  | 0.44 ± 0.08¹| 0.87 ± 0.10  |

¹ Shanghai SampEn is computed on 15-min samples, which collapses the
  fine-scale jitter that drives SampEn at 5 min — the lower value is mostly a
  cadence artefact, not a real complexity difference.

![Variability and complexity panel](figures/variability_metrics.png)

Observations:
- **CV is ~10 pp higher in the simulator** than in either real cohort. This is
  consistent with the wider standard deviation of the pooled distribution
  (89.7 vs 60.8 / 72.3 mg/dL).
- **MAGE** is +44 mg/dL above Ohio and +24 above Shanghai — the simulator’s
  post-meal excursions are larger than either real cohort.
- **CONGA-1h** is ~10 mg/dL higher than Ohio. **CONGA-4h** is ~18 mg/dL
  higher than Ohio in absolute terms but at a smaller relative gap (+23 % vs
  +26 % at 1 h), so the simulator’s excess variability appears at every
  timescale and shrinks only mildly in relative terms with longer windows.
- **MODD is the worst-fit variability metric**: 81 mg/dL vs Ohio 61 and
  Shanghai 53. The simulator’s day-to-day reproducibility is wider than real
  patients’ — the day-1-of-the-week vs day-2-of-the-week variance in meal
  carbs / wake time / behavioural patterns is larger than the real cohorts.
- **Sample entropy matches Ohio almost exactly** (0.87 vs 0.87) — at 5-min
  cadence the simulator produces traces of comparable complexity to real
  CGM.

---

## 6. Temporal dynamics

### 6.1 Autocorrelation

Pooled (mean across records) Pearson autocorrelation at the indicated lag.

| Lag         | OhioT1DM | ShanghaiT1DM | T1DMSIM |
|---|---|---|---|
| 5 min   | 0.995 | (n/a)  | 0.997 |
| 15 min  | 0.969 | 0.984  | 0.979 |
| 30 min  | 0.911 | 0.946  | 0.933 |
| 1 h     | 0.765 | 0.840  | 0.796 |
| 2 h     | 0.484 | 0.606  | 0.511 |
| 4 h     | 0.137 | 0.254  | 0.308 |
| **8 h**     | **−0.004** | **−0.028** | **0.204** |
| **12 h**    | **−0.010** | **−0.050** | **0.178** |
| 24 h    | 0.116 | 0.378  | **0.140** |

![Autocorrelation across lag](figures/acf.png)

Two genuinely different things are happening at the short and long ends:

- **Sub-1-hour ACF matches real CGM tightly.** The simulator’s AR(1) sensor
  noise (ρ = 0.92) and 5-min curve mixing reproduces the Dexcom autocorrelation
  shape almost exactly out to ~ 2 h.
- **Mid-range ACF (4–12 h) is elevated.** Real CGM decorrelates to near
  zero by 8 h. The simulator sits at ρ ≈ 0.20 at 8 h and 0.18 at 12 h —
  still well above real. The underlying cause is the simulator's coupled
  long-memory subsystems (glycogen reservoir, glucotoxicity EMA,
  post-exercise IS boost, meal-HGO rebound) which share state across half
  a day.
- **24 h ACF** is 0.14 in the simulator, very close to Ohio's 0.12.
  Shanghai (0.38) is inflated by short individual records: many patients
  only have ~ 10 days of CGM, so a 24-h lag aligns the daily routine
  deterministically.

### 6.2 Rate-of-change (Δ-BG)

![Δ-BG distribution at native cadence](figures/delta_distribution.png)

At native cadence the simulator’s 5-min Δ-BG distribution is slightly *wider*
than Ohio’s (per-record std ≈ 6.6 vs Ohio 5.5 mg/dL) but tighter than
Shanghai’s 15-min Δ-BG (per-record std ≈ 10.7 mg/dL). The simulator does not
produce the extreme ±30 mg/dL single-step jumps that show up occasionally in
real CGM as sensor recalibration artefacts.

### 6.3 Diurnal pattern (hour-of-day mean ± 1σ across records)

![Hour-of-day mean with ±1σ envelope](figures/diurnal_envelope.png)

Hour-by-hour mean BG (mg/dL):

|   | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | **8** | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Ohio | 149 | 151 | 155 | 160 | 164 | 169 | 173 | 179 | **186** | 178 | 164 | 153 | 154 | 161 | 166 | 165 | 163 | 160 | 162 | 162 | 157 | 158 | 156 | 151 |
| Shanghai | 166 | 164 | 163 | 159 | 156 | 158 | 165 | 169 | **192** | 175 | 144 | 137 | 149 | 143 | 147 | 157 | 166 | 179 | 184 | 175 | 170 | 169 | 168 | 167 |
| Sim | 157 | 143 | 139 | 139 | 143 | 148 | 156 | 166 | 178 | 190 | **193** | 185 | 178 | 180 | 190 | 193 | 187 | 178 | 175 | 184 | 202 | 208 | 197 | 177 |

Differences from Ohio (sim − Ohio) and from Shanghai (sim − Shang) at key hours:

| Hour | Sim − Ohio | Sim − Shanghai | Notes |
|---|---:|---:|---|
| 02–05 (deep night) | **−16 to −21** | **−10 to −24** | Sim overnight is colder than both real cohorts. |
| 08–09 (dawn / breakfast) | −8 to +12 | −14 to +15 | Sim morning peak matches Ohio amplitude; lags Shanghai’s sharper peak by ~ 1 h. |
| 11 (mid-morning post-bolus dip) | +32 | +48 | Sim does not show the Shanghai lunchtime trough. |
| 20–22 (late evening) | +41 to +50 | +29 to +39 | Sim runs ~ 45 mg/dL above Ohio in the late evening — slow dinner clearance tail. |

So the two principal diurnal gaps are (a) the simulator’s **overnight trough
is ~ 20 mg/dL too low** vs Ohio, and (b) its **late-evening peak is ~ 45 mg/dL
too high** vs Ohio. They are mechanically linked: the same residual basal +
delayed-HGO mismatch that overshoots in the evening undershoots overnight as
the bolus tail clears. Shanghai shows different pathologies (sharper morning
spike, distinct lunch dip, no overnight elevation) — its leaner cohort and
mix of regular human insulin produces a qualitatively different diurnal shape
that the simulator does not target.

![Weekday × hour heatmap](figures/weekday_heatmap.png)

The weekday heatmap shows weekend-vs-weekday structure most clearly in the
simulator (where it is parameterised explicitly via `WEEKEND_*` constants);
Ohio shows the same pattern more faintly (real adults shift breakfast on
weekends); Shanghai is too short to resolve a stable weekday × hour cell.

---

## 7. Excursion-level dynamics

### 7.1 Episode counts and durations

Per-record means ± std.

| Metric | OhioT1DM | ShanghaiT1DM | T1DMSIM |
|---|---|---|---|
| Hypo (<70) episodes / day      | 0.81 ± 0.40 | 1.02 ± 0.74 | **1.92 ± 0.86** |
| Severe-hypo (<54) eps / day   | 0.20 ± 0.20 | 0.51 ± 0.47 | 0.22 ± 0.15 |
| Hyper (>180) episodes / day   | 2.61 ± 0.26 | 1.87 ± 0.71 | 1.97 ± 0.32 |
| Severe-hyper (>250) eps / day | 1.06 ± 0.38 | 1.12 ± 0.68 | 1.33 ± 0.51 |
| Hypo median duration (min)    | 33.3 | 69.4 | 36.7 |
| Hypo p90 duration (min)       | 89.8 | 179.3 | **63.0** |
| Hyper median duration (min)   | 131.3 | 213.3 | 161.2 |
| Hyper p90 duration (min)      | 422.8 | 622.6 | **621.4** |

![Episode duration boxplots](figures/episode_durations.png)

Key findings:

- **The simulator has ~2.4× more hypo episodes / day than Ohio** (1.92 vs
  0.81). Each individual sim hypo is *shorter* (p90 63 min vs Ohio 90 min)
  because of the severe-hypo rescue + skill-scaled correction grams in
  `simulator.py`. So the simulator is recovering aggressively from many
  small hypos that real adult patients on closed-loop pumps simply do not
  have. The total fraction of time below 70 (TBR1+TBR2 ≈ 5.5 %) is
  therefore higher than Ohio (3.3 %) but the maximum episode is shorter.
- **Hyper p90 (621 min ≈ 10.4 h) matches Shanghai (623 min) and is ~47 %
  longer than Ohio (423 min)**. A handful of synthetic seeds settle into
  multi-day hyper stretches that are clinically plausible but rare in the
  US Ohio cohort.
- Counts of severe-hypo / severe-hyper episodes per day are close to both
  real cohorts; the simulator does not over-produce dangerous episodes —
  it over-produces *mild* hypo events.

### 7.2 Hypo recovery time

![Hypo recovery time from BG<70 to BG≥80](figures/recovery_time.png)

Time from the first sub-70 sample to the next ≥ 80 sample:

| Cohort | n events | median (min) | p75 (min) | p90 (min) | max (min) |
|---|---:|---:|---:|---:|---:|
| Ohio     | 284   | 50  | 81  | 134 | 295 |
| Shanghai | 157   | 90  | 195 | 300 | 555 |
| Sim      | 4,157 | 40  | 50  | 70  | 375 |

The simulator catches and corrects hypos *faster* than Ohio (median 40 vs
50 min) and much faster than Shanghai (90 min). Its p90 and p99 are well
below either real cohort (sim p99 = 195 min vs Ohio 216 min), although a
small number of rare seeds produce slightly longer single-event maxima
(Sim max 375 min vs Ohio 295 min — about 54 events over 180 min out of
4,157). The fast median recovery is the behaviour enforced by the
`SEVERE_HYPO_THRESHOLD`-triggered ≥14 g rescue + follow-up snack noted in
`CLAUDE.md`.

---

## 8. Per-record (per-patient) heterogeneity

![Per-record TIR vs TBR scatter](figures/per_patient_scatter.png)

| Cohort | TIR IQR (pp) | TIR min–max (pp) | Mean-BG std across records |
|---|---:|---|---:|
| Ohio     |  9.3 | 40.3 – 71.7 | 16.2 |
| Shanghai | 23.1 | 32.1 – 77.3 | 31.0 |
| Sim      | 21.3 | 22.3 – 73.9 | 30.5 |

The simulator’s **cross-record spread (IQR of TIR, std of mean BG across
records) closely matches Shanghai**. The Ohio cohort is six adults from a
single study site and is unrepresentative of the real T1D population’s
heterogeneity; the simulator’s 30 seeds span a similar behavioural
range — its mean-BG std across records (31 mg/dL) is identical to
Shanghai’s, and its TIR IQR (21 pp) is close to Shanghai’s (23 pp).
Whether this is a defect depends on the downstream use: for training a
transformer on T1D dynamics, *more* heterogeneity is desirable; for
matching the Ohio CGM distribution exactly, the simulator overshoots.

A handful of high-LBGI / high-HBGI seeds are visible in the scatter as outlier
records sitting outside the Ohio cluster. They are the same seeds that drive
the heavy right-tail in `figures/pdf_pooled.png` and the +1.20 excess kurtosis.

![LBGI and HBGI per-record boxplots](figures/risk_indices.png)

---

## 9. Synthesis — what is matched, what is not

### 9.1 Where the simulator agrees with real CGM

| Match | Sim | Ohio | Shanghai | Verdict |
|---|---|---|---|---|
| TIR (70–180) | 57.4 | 60.5 | 54.7 | within 3 pp of Ohio, 3 pp of Shanghai |
| **Median BG** | 150.6 | 155.2 | 156.6 | within 6 mg/dL of either |
| Severe-hypo eps / day | 0.22 | 0.20 | 0.51 | matches Ohio |
| TBR2 (<54) | 0.68 % | 0.73 % | 2.79 % | matches Ohio |
| TBR1 (54–70) | 4.82 % | 2.57 % | 4.72 % | matches Shanghai |
| Hyper median duration | 161 min | 131 min | 213 min | between |
| Hyper p90 duration | 621 min | 423 min | 623 min | matches Shanghai |
| Sample entropy | 0.87 | 0.87 | 0.44¹ | matches Ohio at 5-min cadence |
| 5-min — 2-h ACF | matches Ohio within 0.05 across all four sub-2h lags |
| **24 h ACF** | 0.14 | 0.12 | 0.38 | matches Ohio (Shanghai inflated by short records) |
| Per-record heterogeneity | wider TIR IQR than Ohio | — | similar to Shanghai | desirable for training |
| Hypo recovery median | 40 min | 50 min | 90 min | faster than Ohio |

### 9.2 Where the simulator diverges from real CGM

| Gap | Sim | Ohio | Shanghai | Magnitude / cause |
|---|---|---|---|---|
| **Mean BG** | 174.4 | 162.1 | 164.7 | **+10 to +12 mg/dL** vs both real cohorts |
| **GMI** | 7.48 | 7.19 | 7.22 | +0.26 vs both |
| **Pooled std** | 89.7 | 60.8 | 72.3 | +28.9 mg/dL vs Ohio; CV 47.0 % vs 36.2 % |
| **Skewness / kurtosis** | 1.18 / +1.20 | 0.58 / +0.15 | 0.51 / −0.14 | sim distribution is right-skewed and leptokurtic; real cohorts are nearly symmetric |
| **p99** | 465 mg/dL | 326 | 349 | extreme hyper outliers extend ~115 mg/dL beyond either real cohort |
| **TAR2 (>250)** | 18.3 % | 8.9 % | 12.6 % | +9.4 pp vs Ohio, +5.7 pp vs Shanghai |
| **TAR1 (180–250)** | 18.8 % | 27.4 % | 25.1 % | sim under-fills the broad mid-hyper band |
| **Hypo episodes / day** | 1.92 | 0.81 | 1.02 | ~2.4× Ohio rate (offset by shorter duration each) |
| **Overnight (02–05) mean** | 139–148 | 155–164 | 156–159 | sim runs ~20 mg/dL too low overnight |
| **Late-evening (20–22) mean** | 197–208 | 156–162 | 168–175 | sim runs ~45 mg/dL above Ohio late evening |
| **Mid-range ACF (4–12 h)** | 0.18–0.31 | ≈ 0 | ≈ 0 | sim BG is too autocorrelated at half-day timescales |
| **MODD** | 81 mg/dL | 61 | 53 | sim day-to-day reproducibility worse than real patients |
| **CONGA-1h** | 50 | 39 | 34 | sim short-timescale variability +26 % over Ohio |
| **Distribution-distance** | KS 0.10, W₁ 19 mg/dL vs Ohio | — | KS 0.06, W₁ 10 mg/dL inter-real | sim ≈ 2× the inter-real distance |

¹ Shanghai sample entropy is depressed by 15-min cadence — not a real cohort
  difference.

---

## 10. Limitations of this comparison

- **Cohort size.** Ohio (n = 6) and Shanghai (n = 16) are small enough that
  cohort means have non-trivial sampling error; the “real” distribution should
  be taken as a band, not a point. With 30 simulator seeds the sim cohort is
  intentionally larger to bound its own sampling error tightly.
- **Cadence asymmetry.** Shanghai’s 15-min cadence collapses Δ-BG std and
  sample entropy relative to 5-min cohorts. Cross-cadence ACF below 30 min is
  not directly comparable.
- **No glucose-controller benchmark.** The simulator output is compared to two
  real human cohorts but not to UVA/Padova `simglucose` here (kept in a
  separate analysis at `/tmp/compare_uvapadova.py`; see
  `reports/stats.json` for raw numbers if needed).
- **No external behaviour event matching.** Meal and bolus event distributions
  exist in both Ohio XML and Shanghai sheets, but this report compares CGM
  output only — not the carb-bolus pairing distribution, time-to-meal-peak
  alignment, or exercise/sleep co-occurrence. Those are out of scope here.
- **Sample entropy subsampling.** Records longer than 2,500 points are
  subsampled with `np.random.default_rng(0)` so the metric is reproducible but
  is a Monte-Carlo estimate, not the exact value over the full trace.

---

## 11. Reproduction

```bash
# regenerates reports/stats.json + reports/figures/*.png
python reports/build_report.py
```

`scripts/compare_all_datasets.py` is reused for the dataset loaders and grid
regularisation. `OhioT1DM/` and `ShanghaiT1DM/` must be placed at the repo root
(both gitignored, both subject to data-use agreements).

Computed numbers in this report come from one specific run of
`build_report.py` (30 seeds, 70 days each, 24 h warm-up discarded). Re-running
will reproduce them exactly because the simulator is seed-deterministic; the
real-data side is fixed.
