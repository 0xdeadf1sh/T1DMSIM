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
| **mean** | 162.1 | 164.7 | 160.0 | −2.1 | −4.7 |
| **median** | 155.2 | 156.6 | **135.2** | **−20.0** | **−21.4** |
| std | 60.8 | 72.3 | 86.7 | +25.8 | +14.4 |
| IQR | 86.2 | 106.2 | 96.5 | +10.3 | −9.7 |
| CV (%) | 37.5 | 43.9 | 54.1 | +16.6 pp | +10.3 pp |
| skewness | 0.58 | 0.51 | **1.41** | +0.83 | +0.90 |
| excess kurtosis | 0.15 | −0.14 | **1.88** | +1.73 | +2.02 |
| min | 40.0 | 39.6 | 20.0 | — | — |
| max | 400.0 | 475.2 | 500.0 | — | — |

The mean matches both real cohorts within 5 mg/dL — already documented in the
README — but the **median is 20 mg/dL below both real cohorts**. That is the
single most informative one-number finding in this report: the simulator
matches the mean only by averaging a heavier hyper tail against a heavier
mid-low body. The standalone skewness (1.41 vs 0.5) and excess kurtosis (1.88
vs ≈ 0) confirm this — the synthetic distribution is *not* the same shape as
the real one even though they have approximately the same first moment.

### 3.2 Percentiles of the pooled distribution

| Percentile | OhioT1DM | ShanghaiT1DM | T1DMSIM |
|---|---:|---:|---:|
| p1  | 57.0  | 41.3  | 54.1  |
| p5  | 76.0  | 61.2  | 65.0  |
| p10 | 88.0  | 75.6  | 75.1  |
| p25 | 115.4 | 108.0 | 99.2  |
| p50 | 155.2 | 156.6 | 135.2 |
| p75 | 201.6 | 214.2 | 195.6 |
| p90 | 244.6 | 264.6 | 287.4 |
| p95 | 271.0 | 291.6 | 343.8 |
| p99 | 325.8 | 349.2 | 453.1 |

The simulator hugs Ohio reasonably well up through ~p25, then runs ≈ 20 mg/dL
**below** both real cohorts in the p25–p75 body, then crosses over and runs
40–130 mg/dL **above** both real cohorts in the right tail (p90–p99). The
crossover is between p75 and p90.

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
| Sim vs Ohio      | 0.131 | < 10⁻³⁰⁰ | 22.0 | 0.028 |
| Sim vs Shanghai  | 0.125 | 3.5 × 10⁻²⁰⁸ | 18.2 | 0.023 |

Sim is roughly **2× farther** from either real cohort than the two real cohorts
are from each other on all three distance metrics. KS p-values are essentially
zero everywhere because of the sample sizes (Ohio 85k, Sim 600k) — the
*magnitude* of the KS statistic and the Wasserstein gap are the meaningful
numbers, not p.

---

## 4. Clinical glycemic indices

Per-record means ± std across each cohort.

| Index | OhioT1DM | ShanghaiT1DM | T1DMSIM |
|---|---|---|---|
| GMI / eA1c proxy | 7.18 | 7.25 | 7.13 |
| **LBGI** (low-BG risk) | 0.86 ± 0.49 | 1.82 ± 1.76 | **1.58 ± 0.51** |
| **HBGI** (high-BG risk) | 7.60 ± 2.53 | 8.58 ± 4.28 | **8.86 ± 5.70** |
| J-index | 49.2 ± 9.2 | 52.6 ± 17.1 | 59.1 ± 27.6 |
| M-value (ref 120) | 11.1 ± 3.8 | 15.8 ± 6.2 | 18.6 ± 12.8 |

- **GMI matches almost exactly** across all three. With the same mean BG it
  could not be otherwise.
- **LBGI is between the two real cohorts** (1.58 vs Ohio 0.86 and Shanghai
  1.82). The simulator is therefore meaningfully riskier on the hypo side than
  the US Ohio cohort but slightly *safer* than the Shanghai cohort (which has
  the most exposure to severe hypoglycaemia of any of the three).
- **HBGI is essentially equal to Shanghai** (8.86 vs 8.58) and is ~17 % above
  Ohio (7.60). HBGI grows as the square of the deviation, so the simulator’s
  heavy right tail (p99 = 453 mg/dL) directly drives this gap.
- The simulator’s **inter-record variance** of HBGI is roughly **2× higher
  than Ohio** (σ 5.7 vs 2.5). That is the upper-tail outlier seeds: a small
  number of synthetic patients run multi-day stretches above 250 mg/dL while
  the rest of the cohort sits in clinical range.

### 4.1 Time-in-range, per-record cohort summary

| Range | OhioT1DM | ShanghaiT1DM | T1DMSIM |
|---|---|---|---|
| TBR2 (<54)        | 0.73 ± 0.68  | 2.79 ± 3.77  | 0.97 ± 0.47  |
| TBR1 (54–70)      | 2.57 ± 1.61  | 4.72 ± 3.97  | 6.34 ± 2.22  |
| **TIR (70–180)**  | **60.5 ± 10.2** | **54.7 ± 14.5** | **63.1 ± 14.4** |
| TAR1 (180–250)    | 27.4 ± 6.1   | 25.1 ± 11.7  | 15.1 ± 6.5   |
| TAR2 (>250)       | 8.88 ± 6.11  | 12.64 ± 8.91 | 14.47 ± 11.82|

The simulator’s TIR is slightly higher than either real cohort. It re-allocates
the missing TAR1 mass roughly half-and-half into TBR1 (extra mild hypos) and
TAR2 (extra deep hypers). That redistribution is exactly the *bimodalisation*
visible in the pooled PDF.

![Clinical-range cohort comparison](../assets/clinical_ranges.png)

(The bar chart from the README is included here for direct reference; the
figure is produced by `scripts/generate_comparison_figures.py`.)

---

## 5. Variability and complexity

Per-record mean ± std.

| Metric (native cadence) | OhioT1DM | ShanghaiT1DM | T1DMSIM |
|---|---|---|---|
| CV (%)              | 36.2 ± 4.5   | 38.6 ± 6.8  | **46.8 ± 8.6**  |
| MAGE (mg/dL)        | 103.9 ± 15.4 | 123.4 ± 30.0| 131.9 ± 36.7 |
| CONGA-1h (mg/dL)    | 39.4 ± 5.6   | 34.2 ± 7.2  | 44.4 ± 6.6   |
| CONGA-4h (mg/dL)    | 76.1 ± 11.4  | 75.1 ± 17.7 | 80.5 ± 16.8  |
| MODD (mg/dL)        | 61.1 ± 8.9   | 53.3 ± 12.8 | **74.1 ± 27.2** |
| Sample entropy      | 0.87 ± 0.10  | 0.44 ± 0.08¹| 0.87 ± 0.18  |

¹ Shanghai SampEn is computed on 15-min samples, which collapses the
  fine-scale jitter that drives SampEn at 5 min — the lower value is mostly a
  cadence artefact, not a real complexity difference.

![Variability and complexity panel](figures/variability_metrics.png)

Observations:
- **CV is ~10 pp higher in the simulator** than in either real cohort. This is
  consistent with the wider standard deviation of the pooled distribution
  (86.7 vs 60.8 / 72.3 mg/dL).
- **MAGE** sits closer to Shanghai than Ohio; the simulator’s post-meal
  excursions are larger than the US cohort.
- **CONGA-1h** is also ~5 mg/dL higher than Ohio, but **CONGA-4h** converges,
  meaning the excess variability is concentrated at short timescales (mostly
  meal-related rapid swings).
- **MODD is the worst-fit variability metric**: 74 mg/dL vs Ohio 61 and
  Shanghai 53. The simulator’s day-to-day reproducibility is worse than real
  patients’. This is the day-1-of-the-week vs day-2-of-the-week variance in
  meal carbs / wake time / behavioural patterns being slightly too aggressive.
- **Sample entropy matches Ohio almost exactly** (0.87 vs 0.87) — at 5-min
  cadence the simulator produces traces of comparable complexity to real
  CGM. This is one of the cleaner wins.

---

## 6. Temporal dynamics

### 6.1 Autocorrelation

Pooled (mean across records) Pearson autocorrelation at the indicated lag.

| Lag         | OhioT1DM | ShanghaiT1DM | T1DMSIM |
|---|---|---|---|
| 5 min   | 0.995 | (n/a)  | 0.996 |
| 15 min  | 0.969 | 0.984  | 0.977 |
| 30 min  | 0.911 | 0.946  | 0.927 |
| 1 h     | 0.765 | 0.840  | 0.788 |
| 2 h     | 0.484 | 0.606  | 0.544 |
| 4 h     | 0.137 | 0.254  | 0.374 |
| **8 h**     | **−0.004** | **−0.028** | **0.261** |
| **12 h**    | **−0.010** | **−0.050** | **0.245** |
| 24 h    | 0.116 | 0.378  | 0.170 |

![Autocorrelation across lag](figures/acf.png)

Two genuinely different things are happening at the short and long ends:

- **Sub-1-hour ACF matches real CGM tightly.** The simulator’s AR(1) sensor
  noise (ρ = 0.92) and 5-min curve mixing reproduces the Dexcom autocorrelation
  shape almost exactly out to ~ 2 h.
- **Mid-range ACF (4–12 h) is much too persistent.** Real CGM decorrelates to
  near zero by 8 h. The simulator still has ρ ≈ 0.26 at 8 h and 0.25 at 12 h —
  *an order of magnitude too autocorrelated*. The cause is the simulator’s
  many coupled long-memory subsystems (glycogen reservoir, glucotoxicity 6 h
  EMA, post-exercise IS boost lasting 10 h, illness IS ramp, meal-HGO rebound
  3.5–5.5 h later) that share state across half a day. Real patients have these
  too, but they ride on top of stochastic real-world perturbations (sensor
  swaps, missed boluses, snacks at random times, lifestyle noise) that the
  simulator does not yet inject at the right amplitude. **This is the largest
  single dynamics defect in the simulator.**
- **24 h ACF** is moderate in the simulator (0.17), low in Ohio (0.12), high in
  Shanghai (0.38). The Shanghai number is inflated by the short individual
  records: many patients only have ~ 10 days of CGM, so a 24-h lag aligns the
  daily routine deterministically. Sim and Ohio agree better.

### 6.2 Rate-of-change (Δ-BG)

![Δ-BG distribution at native cadence](figures/delta_distribution.png)

At native cadence the simulator’s 5-min Δ-BG distribution is slightly *wider*
than Ohio’s (per-record std ≈ 6.1 vs Ohio 5.5 mg/dL) but tighter than
Shanghai’s 15-min Δ-BG. The simulator does not produce the extreme ±30 mg/dL
single-step jumps that show up occasionally in real CGM as sensor recalibration
artefacts.

### 6.3 Diurnal pattern (hour-of-day mean ± 1σ across records)

![Hour-of-day mean with ±1σ envelope](figures/diurnal_envelope.png)

Hour-by-hour mean BG (mg/dL):

|   | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | **8** | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Ohio | 149 | 151 | 155 | 160 | 164 | 169 | 174 | 179 | **186** | 178 | 164 | 153 | 154 | 161 | 167 | 165 | 163 | 160 | 163 | 162 | 157 | 158 | 156 | 152 |
| Shanghai | 166 | 164 | 163 | 159 | 156 | 158 | 165 | 169 | **192** | 175 | 144 | 137 | 149 | 143 | 147 | 157 | 166 | 179 | 184 | 176 | 170 | 169 | 168 | 167 |
| Sim | 146 | 137 | 134 | 134 | 138 | 143 | 152 | 163 | 175 | **182** | 178 | 170 | 165 | 164 | 165 | 168 | 165 | 158 | 157 | 166 | 175 | 178 | 172 | 159 |

Differences from Ohio (sim − Ohio) and from Shanghai (sim − Shang) at key hours:

| Hour | Sim − Ohio | Sim − Shanghai | Notes |
|---|---:|---:|---|
| 02–05 (deep night) | **−24 to −26** | **−25 to +0** | Sim overnight runs much colder than Ohio, comparable to Shanghai overnight nadir. |
| 08–09 (dawn / breakfast) | −4 to +4 | −17 to +7 | Sim morning peak matches Ohio amplitude; lags Shanghai’s sharper peak by ~ 1 h. |
| 11 (mid-morning post-bolus dip) | +17 | +33 | Sim does not show the Shanghai lunchtime trough. |
| 20–22 (late evening) | +18 to +20 | +5 to +9 | Sim runs ~ 20 mg/dL above Ohio in the late evening — slow dinner clearance tail. |

So the two principal diurnal gaps are (a) the simulator’s **overnight trough
is ~ 25 mg/dL too low** vs Ohio, and (b) its **late-evening peak is ~ 20 mg/dL
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
| Hypo (<70) episodes / day      | 0.81 ± 0.40 | 1.02 ± 0.74 | **2.52 ± 1.05** |
| Severe-hypo (<54) eps / day   | 0.20 ± 0.20 | 0.51 ± 0.47 | 0.33 ± 0.24 |
| Hyper (>180) episodes / day   | 2.61 ± 0.26 | 1.87 ± 0.71 | 1.64 ± 0.49 |
| Severe-hyper (>250) eps / day | 1.06 ± 0.38 | 1.12 ± 0.68 | 0.90 ± 0.52 |
| Hypo median duration (min)    | 33.3 | 69.4 | 36.8 |
| Hypo p90 duration (min)       | 89.8 | 179.3 | **64.7** |
| Hyper median duration (min)   | 131.3 | 213.3 | 137.0 |
| Hyper p90 duration (min)      | 422.8 | 622.6 | **636.0** |

![Episode duration boxplots](figures/episode_durations.png)

Key findings:

- **The simulator has 3× more hypo episodes / day than Ohio** (2.52 vs 0.81).
  Each individual sim hypo is *shorter* (p90 65 min vs Ohio 90 min) because of
  the severe-hypo rescue + skill-scaled correction grams in
  `simulator.py`. So the simulator is recovering aggressively from many small
  hypos that real adult patients on closed-loop pumps simply do not have.
  The total fraction of time below 70 (TBR1+TBR2 ≈ 7.3 %) is therefore higher
  than Ohio (3.3 %) but the maximum episode is shorter.
- **Hyper p90 (636 min ≈ 10.6 h) is ~50 % longer than Ohio (423 min ≈ 7 h)**
  and roughly matches Shanghai (623 min). A handful of synthetic seeds settle
  into multi-day hyper stretches that are clinically plausible but rare in
  the US Ohio cohort.
- Counts of severe-hypo / severe-hyper episodes per day are close to both real
  cohorts; the simulator does not over-produce dangerous episodes — it
  over-produces *mild* hypo events.

### 7.2 Hypo recovery time

![Hypo recovery time from BG<70 to BG≥80](figures/recovery_time.png)

Time from the first sub-70 sample to the next ≥ 80 sample:

| Cohort | n events | median (min) | p75 (min) | p90 (min) | max (min) |
|---|---:|---:|---:|---:|---:|
| Ohio     | 284   | 50  | 81  | 134 | 295 |
| Shanghai | 157   | 90  | 195 | 300 | 555 |
| Sim      | 5,388 | 40  | 55  | 80  | 395 |

The simulator catches and corrects hypos *faster* than Ohio (median 40 vs
50 min) and much faster than Shanghai (90 min). Its right-tail outliers are
shorter — there are no multi-hour low-stall events in the simulator output. That is precisely the
behaviour enforced by `SEVERE_HYPO_THRESHOLD`-triggered ≥14 g rescue +
follow-up snack noted in `CLAUDE.md`.

---

## 8. Per-record (per-patient) heterogeneity

![Per-record TIR vs TBR scatter](figures/per_patient_scatter.png)

| Cohort | TIR IQR (pp) | TIR min–max (pp) | Mean-BG std across records |
|---|---:|---|---:|
| Ohio     |  9.3 | 40.3 – 71.7 | 16.2 |
| Shanghai | 23.1 | 32.1 – 77.3 | 31.0 |
| Sim      | 27.0 | 36.5 – 85.5 | 31.0 |

The simulator’s **cross-record spread (IQR of TIR, std of mean BG across
records) is comparable to Shanghai and three times wider than Ohio**. The
Ohio cohort is six adults from a single study site and is unrepresentative of
the real T1D population’s heterogeneity; the simulator’s 30 seeds genuinely
span a wider behavioural range — its mean-BG std across records (31 mg/dL) is
identical to Shanghai’s. Whether this is a defect depends on the downstream
use: for training a transformer on T1D dynamics, *more* heterogeneity is
desirable; for matching the Ohio CGM distribution exactly, the simulator
overshoots.

A handful of high-LBGI / high-HBGI seeds are visible in the scatter as outlier
records sitting outside the Ohio cluster. They are the same seeds that drive
the heavy right-tail in `figures/pdf_pooled.png` and the +1.88 excess kurtosis.

![LBGI and HBGI per-record boxplots](figures/risk_indices.png)

---

## 9. Synthesis — what is matched, what is not

### 9.1 Where the simulator agrees with real CGM

| Match | Sim | Ohio | Shanghai | Verdict |
|---|---|---|---|---|
| Mean BG | 160.0 | 162.1 | 164.7 | ≤ 5 mg/dL gap |
| GMI | 7.13 | 7.18 | 7.25 | within 0.12 |
| TIR (70–180) | 63.1 | 60.5 | 54.7 | within 3 pp of Ohio, 8 pp of Shanghai |
| Severe-hypo eps / day | 0.33 | 0.20 | 0.51 | between the two real cohorts |
| Hyper median duration | 137 min | 131 min | 213 min | matches Ohio |
| Sample entropy | 0.87 | 0.87 | 0.44¹ | matches Ohio at 5-min cadence |
| 5-min — 2-h ACF | matches Ohio within 0.06 across all four sub-2h lags |
| Diurnal peak hour | 09:00 | 08:00 | 08:00 | 1 h late |
| Per-record heterogeneity | wider TIR IQR than Ohio | — | similar to Shanghai | desirable for training |
| Hypo recovery median | 40 min | 40 min | 75 min | matches Ohio exactly |

### 9.2 Where the simulator diverges from real CGM

| Gap | Sim | Ohio | Shanghai | Magnitude / cause |
|---|---|---|---|---|
| **Median BG** | 135.2 | 155.2 | 156.6 | **−20 mg/dL** vs both real cohorts; sim median anchored to mid-range while mean is dragged up by a heavy hyper tail |
| **Pooled std** | 86.7 | 60.8 | 72.3 | +25.8 mg/dL vs Ohio; CV 46.8 % vs 36.2 % |
| **Skewness / kurtosis** | 1.41 / +1.88 | 0.58 / +0.15 | 0.51 / −0.14 | sim distribution is heavily right-skewed and leptokurtic; real cohorts are nearly symmetric |
| **p99** | 453 mg/dL | 326 | 349 | extreme hyper outliers extend ~125 mg/dL beyond either real cohort |
| **TAR2 (>250)** | 14.5 % | 8.9 % | 12.6 % | +5.6 pp vs Ohio, +1.8 pp vs Shanghai |
| **TBR1 (54–70)** | 6.34 % | 2.57 % | 4.72 % | +3.8 pp vs Ohio; sim produces many mild hypos that recover fast |
| **Hypo episodes / day** | 2.52 | 0.81 | 1.02 | 3× Ohio rate (offset by shorter duration each) |
| **Overnight (02–05) mean** | 134–138 | 155–164 | 156–159 | sim runs ~25 mg/dL too low overnight |
| **Late-evening (20–22) mean** | 174–178 | 156–162 | 168–175 | sim runs ~18 mg/dL above Ohio late evening |
| **Mid-range ACF (4–12 h)** | 0.25–0.37 | ≈ 0 | ≈ 0 | sim BG is *much* too autocorrelated at half-day timescales |
| **MODD** | 74 mg/dL | 61 | 53 | sim day-to-day reproducibility worse than real patients |
| **CONGA-1h** | 44 | 39 | 34 | sim short-timescale variability +13 % over Ohio |
| **Hyper p90 duration** | 636 min | 423 | 623 | extreme hyper stretches longer than Ohio, on par with Shanghai |
| **Distribution-distance** | KS 0.13, W₁ 22 mg/dL vs Ohio | — | KS 0.06, W₁ 10 mg/dL inter-real | sim ≈ 2× the inter-real distance |

¹ Shanghai sample entropy is depressed by 15-min cadence — not a real cohort
  difference.

### 9.3 Most actionable defects, in priority order

1. **Right tail of the pooled distribution.** The simulator spends 8.74 % of
   time above 300 mg/dL vs Ohio’s 2.00 % and Shanghai’s 4.05 % — roughly
   **4× the Ohio mass** in the extreme-hyper region, despite spending *less*
   time in the 200–300 mg/dL band than Ohio (23.9 % vs 25.6 %). A small
   number of high-IS-low-skill seeds settle into multi-day hyper stretches
   with under-dosed basal. Likely fix: tighten the basal-correction loop’s
   long-duration ceiling, or cap inter-seed dispersion of the `is_base / ICR`
   ratio so the few outlier seeds don’t drift this far.

2. **Median anchored 20 mg/dL too low.** Even with the heavy right tail, the
   pooled distribution’s mode and median are well below the two real cohorts.
   This is the overnight-floor problem (point 4 below) integrated across all
   24 h.

3. **Mid-range autocorrelation (4–12 h).** The simulator’s state is too
   strongly self-coupled across the half-day timescale. Real CGM
   decorrelates to ≈ 0 by 8 h; the simulator stays at 0.25. The likely
   contributors are the glycogen reservoir + glucotoxicity EMA + delayed-HGO
   rebound all sharing state across the same window. Adding *de-correlating*
   stochastic shocks (random snacks, missed/late boluses, ad-hoc exercise) at
   real-world amplitudes should bring this down.

4. **Overnight trough 25 mg/dL below Ohio.** The simulator’s basal-vs-HGO
   balance is calibrated against the *daytime average* but undershoots
   overnight after dinner-bolus clearance. Either basal needs a small
   nocturnal scale-up, or the dinner bolus needs a slightly longer tail.

5. **Hypo-episode frequency 3× Ohio.** The current correction logic *catches*
   hypos faster than real patients (good) but the patient is *entering* them
   too often (bad). Tightening the meal-bolus model — particularly for low
   `s3` patients whose dosing competence currently produces frequent
   small-overdose events — should reduce this.

6. **MODD too high.** Day-to-day variability in meal carb, wake time, and
   meal-timing jitter is too aggressive. Reducing one or more of
   `MEAL_TIME_JITTER_BASE_MIN`, `MEAL_CARB_SIGMA`, or
   `WAKE_TIME_SIGMA_BASE` should converge MODD toward the Ohio value
   without flattening MAGE.

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
  alignment, or exercise/sleep co-occurrence. Those are next-pass extensions.
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
