# Which simulator better resembles real-world CGM?

## Population summary

| Cohort | Mean BG | GMI | TIR 70–180 | 5-min ΔBG SD |
|---|---|---|---|---|
| **Real (pooled)** | 155 | 7.01 | 68% | 6.00 |
| · Ohio | 162 | 7.19 | 61% | 5.89 |
| · Shanghai (15-min) | 164 | 7.22 | 55% | — |
| · AZT1D | 147 | 6.83 | 78% | 6.02 |
| **T1DMSIM** | 163 | 7.20 | 61% | 5.51 |
| **UVA/Padova** | 120 | 6.19 | 93% | 4.01 |

## Distance to real (lower is closer)

| Wasserstein-1 | T1DMSIM | UVA/Padova | real-vs-real floor |
|---|---|---|---|
| BG distribution | 13.7 | 30.0 | 18.4 |
| ΔBG distribution | 0.7 | 1.2 | 0.3 |

![BG distribution](figures/realism_pdf.png)
![BG CDF](figures/realism_cdf.png)
![Distance to real](figures/realism_distance_bars.png)

## Per-metric distance (units of one real-patient SD)

### Marginal & risk metrics (sampling-invariant; all three real cohorts)

| Metric | Real | T1DMSIM | UVA/Padova | ours (SD) | UVA (SD) | closer |
|---|---|---|---|---|---|---|
| Mean BG | 154.6 | 162.6 | 120.5 | 0.34 | 1.44 | T1DMSIM |
| CV | 33.6 | 36.6 | 22.9 | 0.46 | 1.59 | T1DMSIM |
| TIR 70-180 | 67.8 | 60.9 | 93.5 | 0.42 | 1.58 | T1DMSIM |
| TBR 54-70 | 2.6 | 3.0 | 2.4 | 0.15 | 0.07 | UVA/Padova |
| TAR 180-250 | 21.0 | 26.4 | 3.0 | 0.54 | 1.80 | T1DMSIM |
| 10th pct | 92.2 | 90.9 | 85.9 | 0.08 | 0.39 | T1DMSIM |
| 90th pct | 224.6 | 244.5 | 157.8 | 0.53 | 1.79 | T1DMSIM |
| LBGI | 1.0 | 0.8 | 1.1 | 0.16 | 0.09 | UVA/Padova |
| HBGI | 6.4 | 7.6 | 1.3 | 0.35 | 1.39 | T1DMSIM |

| Summary | T1DMSIM | UVA/Padova |
|---|---|---|
| Mean normalised error | 0.34 | 1.13 |

### Rate-of-change metrics (cadence-dependent; 5-min cohorts only — the fairer test)

| Metric | Real | T1DMSIM | UVA/Padova | ours (SD) | UVA (SD) | closer |
|---|---|---|---|---|---|---|
| 5-min ΔBG SD | 6.0 | 5.5 | 4.0 | 0.54 | 2.18 | T1DMSIM |
| MAGE | 82.9 | 99.0 | 47.7 | 0.86 | 1.89 | T1DMSIM |
| CONGA-1h | 37.9 | 38.9 | 24.5 | 0.18 | 2.41 | T1DMSIM |
| MODD | 45.8 | 63.2 | 22.1 | 1.53 | 2.08 | T1DMSIM |
| Sample entropy | 0.7 | 0.6 | 1.1 | 0.59 | 2.75 | T1DMSIM |

| Summary | T1DMSIM | UVA/Padova |
|---|---|---|
| Mean normalised error | 0.74 | 2.26 |

![Per-metric distance](figures/realism_metric_error.png)
![ΔBG distribution](figures/realism_delta.png)
![Autocorrelation](figures/realism_acf.png)
