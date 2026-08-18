# Which simulator better resembles real-world CGM?

## Population summary

| Cohort | Mean BG | GMI | TIR 70–180 | 5-min ΔBG SD |
|---|---|---|---|---|
| **Real (pooled)** | 155 | 7.01 | 68% | 6.00 |
| · Ohio | 162 | 7.19 | 61% | 5.89 |
| · Shanghai (15-min) | 164 | 7.22 | 55% | — |
| · AZT1D | 147 | 6.83 | 78% | 6.02 |
| **T1DMSIM** | 161 | 7.16 | 57% | 5.28 |
| **UVA/Padova** | 120 | 6.19 | 93% | 4.01 |

## Distance to real (lower is closer)

| Wasserstein-1 | T1DMSIM | UVA/Padova | real-vs-real floor |
|---|---|---|---|
| BG distribution | 16.7 | 30.0 | 18.4 |
| ΔBG distribution | 0.5 | 1.2 | 0.3 |

![BG distribution](figures/realism_pdf.png)
![BG CDF](figures/realism_cdf.png)
![Distance to real](figures/realism_distance_bars.png)

## Per-metric distance (units of one real-patient SD)

### Marginal & risk metrics (sampling-invariant; all three real cohorts)

| Metric | Real | T1DMSIM | UVA/Padova | ours (SD) | UVA (SD) | closer |
|---|---|---|---|---|---|---|
| Mean BG | 154.6 | 160.8 | 120.5 | 0.26 | 1.44 | T1DMSIM |
| CV | 33.6 | 38.3 | 22.9 | 0.70 | 1.59 | T1DMSIM |
| TIR 70-180 | 67.8 | 57.3 | 93.5 | 0.64 | 1.58 | T1DMSIM |
| TBR 54-70 | 2.6 | 5.0 | 2.4 | 0.81 | 0.07 | UVA/Padova |
| TAR 180-250 | 21.0 | 26.0 | 3.0 | 0.50 | 1.80 | T1DMSIM |
| 10th pct | 92.2 | 87.1 | 85.9 | 0.31 | 0.39 | T1DMSIM |
| 90th pct | 224.6 | 244.9 | 157.8 | 0.54 | 1.79 | T1DMSIM |
| LBGI | 1.0 | 1.7 | 1.1 | 0.57 | 0.09 | UVA/Padova |
| HBGI | 6.4 | 7.9 | 1.3 | 0.42 | 1.39 | T1DMSIM |

| Summary | T1DMSIM | UVA/Padova |
|---|---|---|
| Mean normalised error | 0.53 | 1.13 |

### Rate-of-change metrics (cadence-dependent; 5-min cohorts only — the fairer test)

| Metric | Real | T1DMSIM | UVA/Padova | ours (SD) | UVA (SD) | closer |
|---|---|---|---|---|---|---|
| 5-min ΔBG SD | 6.0 | 5.3 | 4.0 | 0.79 | 2.18 | T1DMSIM |
| MAGE | 82.9 | 111.1 | 47.7 | 1.52 | 1.89 | T1DMSIM |
| CONGA-1h | 37.9 | 35.7 | 24.5 | 0.39 | 2.41 | T1DMSIM |
| MODD | 45.8 | 59.8 | 22.1 | 1.23 | 2.08 | T1DMSIM |
| Sample entropy | 0.7 | 0.5 | 1.1 | 1.35 | 2.75 | T1DMSIM |

| Summary | T1DMSIM | UVA/Padova |
|---|---|---|
| Mean normalised error | 1.06 | 2.26 |

![Per-metric distance](figures/realism_metric_error.png)
![ΔBG distribution](figures/realism_delta.png)
![Autocorrelation](figures/realism_acf.png)
