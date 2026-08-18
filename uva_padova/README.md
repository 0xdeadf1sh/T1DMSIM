# T1DMSIM vs. UVA/Padova 2008 (in-silico reference)

## Distributional comparison (pooled across seeds)

| Metric | T1DMSIM | UVA/Padova | Δ (UVA − ours) |
|---|---|---|---|
| Mean BG (mg/dL) | 165.5 | 168.0 | 2.4 |
| Median BG (mg/dL) | 159.6 | 156.8 | -2.7 |
| SD (mg/dL) | 60.0 | 68.1 | 8.1 |
| CV (%) | 36.7 | 55.5 | 18.8 |
| GMI (%) | 7.27 | 7.33 | 0.06 |
| p10 (mg/dL) | 93.3 | 91.0 | -2.3 |
| p90 (mg/dL) | 249.1 | 261.0 | 11.9 |
| Time <54 % (TBR2) | 1.60 | 13.22 | 11.63 |
| Time 54–70 % (TBR1) | 4.18 | 4.04 | -0.13 |
| Time 70–180 % (TIR) | 54.9 | 43.3 | -11.6 |
| Time 180–250 % (TAR1) | 28.2 | 20.3 | -7.9 |
| Time >250 % (TAR2) | 11.10 | 19.09 | 7.99 |
| LBGI | 1.61 | 56.34 | 54.73 |
| HBGI | 8.70 | 11.91 | 3.21 |
| Hypo episodes/day | 0.98 | 0.52 | -0.46 |
| Hyper episodes/day | 1.49 | 1.08 | -0.42 |
| 5-min ΔBG SD (mg/dL) | 3.37 | 2.64 | -0.73 |

![Pooled BG distribution](figures/pdf_overlay.png)
![Pooled BG CDF](figures/cdf_overlay.png)
![Time-in-range bands](figures/tir_bars.png)
![Diurnal mean BG](figures/diurnal.png)
![Rate-of-change distribution](figures/delta_dist.png)

## Paired-curve agreement

| Quantity | Median | IQR |
|---|---|---|
| RMSE (mg/dL) | 107.8 | 88.8–124.0 |
| Mean abs. diff (mg/dL) | 83.9 | 74.3–108.4 |
| Pearson r | 0.26 | 0.22–0.44 |
| Pearson r (best lag) | 0.29 | 0.23–0.44 |
| Mean-BG drift (mg/dL) | +4.1 | -62.6–+71.1 |
| KS distance | 0.465 | 0.345–0.557 |
| Wasserstein-1 (mg/dL) | 69.0 | 51.5–97.9 |

![Representative paired trace, 72 h](figures/paired_trace.png)
![Representative paired trace, 24 h](figures/paired_trace_zoom.png)
![Bland–Altman](figures/bland_altman.png)
![Per-seed agreement](figures/per_seed_scatter.png)

## Generation speed

| Simulated days | T1DMSIM (s) | UVA/Padova (s) | T1DMSIM ms/day | UVA/Padova ms/day | Speedup |
|---|---|---|---|---|---|
| 1 | 0.004 | 3.69 | 4.23 | 3694 | ×874 |
| 3 | 0.015 | 11.57 | 4.87 | 3856 | ×792 |
| 7 | 0.035 | 27.51 | 5.06 | 3931 | ×777 |
| 14 | 0.077 | 54.67 | 5.50 | 3905 | ×710 |

| Throughput | Value |
|---|---|
| T1DMSIM | 52364 steps/s |
| UVA/Padova | 369 ODE-minutes/s |
| End-to-end speedup at the longest horizon | ×710 |

![Speed benchmark](figures/speed_bench.png)
