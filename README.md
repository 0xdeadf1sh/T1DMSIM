# T1DM Patient Behavior Simulator

A seed-driven simulator for generating synthetic Type 1 Diabetes blood glucose data. Unlike traditional glucose-insulin simulators (e.g., UVA/Padova), this simulator models patient *behavior* as the primary driver of blood sugar outcomes. The simulator generates factor curves -- carbohydrate intake, insulin action, insulin sensitivity, and exercise -- and computes blood sugar as the emergent result of their interactions.

Designed by a T1DM patient, informed by lived experience.

> [!CAUTION]
> **Research and educational use only.** This project is a synthetic-data generator and a behavioral model of Type 1 Diabetes — not a medical device, and not clinically validated. Its output is artificial data, not real patient measurements, and **must not** be used to make medical, diagnostic, or treatment decisions, to calculate or adjust insulin doses, or to guide diabetes management in any way. For medical advice, consult a qualified healthcare professional. The software is provided "as is", without warranty of any kind, and the authors accept no liability for any use.

Screenshot:

![Software Screenshot](screenshots/t1dm_seed42_36h.png)


## Table of contents

- [Motivation](#motivation)
- [Pregenerated Dataset](#pregenerated-dataset)
- [Design Principles](#design-principles)
- [Architecture](#architecture)
- [Blood Sugar Computation](#blood-sugar-computation)
- [Patient Model](#patient-model)
- [Insulin Sensitivity Model](#insulin-sensitivity-model)
- [Behavioral Events](#behavioral-events)
- [Installation and Usage](#installation-and-usage)
- [Visualizer Controls](#visualizer-controls)
- [Parameters](#parameters)
- [Comparison Against Real-World Datasets](#comparison-against-real-world-datasets)
- [Comparison Against the UVA/Padova Simulator](#comparison-against-the-uvapadova-simulator)
- [Testing](#testing)
- [References](#references)
- [License](#license)


## Motivation

Most T1DM simulators model physiology: glucose kinetics, insulin pharmacokinetics, compartmental models. They produce accurate BG traces but require dozens of physiological parameters that are hard to measure and vary between patients.

This simulator takes a different approach. It models the *person*, not the pancreas. The key insight is that most real-world blood sugar variance comes from behavioral decisions -- what the patient eats, when they bolus, how they correct, whether they exercise -- not from subtle physiological differences. By generating diverse behavioral patterns and computing BG as a consequence, the simulator produces training data that teaches a model to predict what patients *do*, with blood sugar as the outcome.

The ultimate goal is to provide a near-unlimited stream of synthetic factor curves that can be used to pretrain ML models for personalized blood sugar prediction, with real patient data reserved for fine-tuning.

## Pregenerated Dataset

There is a pregenerated dataset released to the public domain that is available for download [here](https://drive.google.com/drive/folders/1UhxoJNiUSOwUGybmdxbyspKQYYN9UqOm?usp=sharing).
It contains **100 million** independent simulated T1D patient trajectories, each 55.5 h of post-warmup CGM at 5-min cadence (666 steps; the first 48 h of warmup are discarded before caching). Stored as raw `.npy` memmaps — 8 per-step channels + `icr`, uncompressed.

| Quantity | Value | Basis |
|---|---|---|
| **Patients (trajectories)** | **100,000,000** | exact (`meta.json`) |
| Timesteps / patient | 666 @ 5 min | exact |
| Duration / patient | 55.5 h ≈ 2.31 days | exact |
| **Total CGM readings** | **66.6 billion** (100M × 666) | exact |
| **Total CGM hours** | **5.55 billion h** | exact |
| **Total CGM years** | **≈ 633,000 years** | exact (= patient-years; 231 M patient-days) |
| **Disk footprint** | **2.13 TB** (1.94 TiB) | exact (`du`); 8 × 266.4 GB + 0.4 GB `icr` |
| **Carbs eaten** | **≈ 39,000 tonnes** (3.90×10¹⁰ g) | est. — 390 g/patient, 168.7 g/day |
| **Meals eaten** | **≈ 732 million** | est. — 7.3/patient, 3.16/day |
| **Insulin delivered** | **≈ 9.87 billion U** (~98,700 L of U-100) | est. — 98.7 U/patient, 42.7 U/day |
| └ basal : bolus split | ≈ 62% : 38% | est. (basal-majority, as designed) |
| Boluses | ~one per meal (~3/day) by construction | detector undercounts (overlapping PK tails) |
| Exercise | active 5.2% of time; ~0.96 sessions/patient | est. |
| BG mean | 157 mg/dL | est. |
| Time-in-range 70–180 | 61.5% | est. |
| Time <70 / <54 (hypo) | 6.7% / 0.68% | est. |
| Time >180 / >250 (hyper) | 31.8% / 10.3% | est. |
| ICR (carb ratio) | 8.1 g/U mean (≈3.9–17.6) | est. |

### Per-channel inventory (each 100M × 666)

`bg_observed`, `total_carb`, `total_insulin`, `insulin_resistance`, `hgo`, `total_exercise`, `hour_of_day` (float32, 266.4 GB each); `day` (int32, 266.4 GB); `icr` (float32, 100M, 0.4 GB).

> **Method.** Patient count, durations, CGM-hours and disk are exact (metadata + `du`). Consumption figures are a 0.5% stratified sample (2,000 blocks × 250 contiguous rows = 500,000 patients), extrapolated ×200. Carbs and insulin are integrals of the per-step appearance/delivery rates, so sums read as grams ingested and units dosed. "Meals" are upward threshold-crossings (>1 g/step, 30-min refractory) in the carb-appearance trace, validated against the simulator's ~3.3 meals/day model. The discrete bolus count is unreliable (basal and bolus share one channel with overlapping pharmacokinetics); the units total and basal/bolus split are reliable.

## Design Principles

The simulator is built on several core ideas:

1. Every factor is a curve, not a number. Eating 40g of bread and 40g of orange juice both contribute 40g of carbs, but the absorption curves have different shapes (the juice peaks faster and falls faster). The same applies to rapid-acting vs long-acting insulin.

2. Patient behavior is driven by a latent skill profile. Four correlated skill dimensions (dietary discipline, attentiveness, dosing competence, lifestyle consistency) determine everything about how a patient lives: what they eat, when they eat, how accurately they dose, how quickly they correct, whether they exercise.

3. The liver is an insulin-suppressed feeding session. Hepatic glucose output (HGO) is a steady stream of "food" entering the bloodstream, throttled down by a Hill function of EMA-smoothed plasma insulin. A finite hepatic glycogen reservoir gates the glycogenolysis fraction, and large meals schedule a delayed HGO rebound 3.5-5.5h later (the mechanism behind nocturnal hyperglycemia after a big dinner). Basal insulin exists to counteract baseline HGO; the ideal basal dose is anchored to `HGO_base × 24h × (body_weight_kg / BODY_WEIGHT_MEAN_KG) × is_base / ICR`.

4. Exercise is negative food. Walking, for example, pulls glucose out of the bloodstream into muscle cells. Modeling this as a negative carb-equivalent curve is a pragmatic simplification that works well for aerobic exercise. Additionally, exercise increases insulin sensitivity for `EXERCISE_IS_DURATION_HOURS` (6h) afterward, modeled as a time-limited IS reduction.

5. Everything is seed-driven. A single integer seed determines the patient's personality, physiology, daily schedule, meal choices, insulin doses, exercise patterns, illness events, and random noise. Same seed, same simulation, always.


## Architecture

The simulator consists of two files:

`simulator.py` contains the core engine. All tunable parameters are defined as uppercase constants at the top of the file (approximately 270 parameters). The `T1DMSimulator` class exposes a `generate()` method that advances the simulation by one 5-minute time step and returns all factor values and the resulting BG. This is analogous to `rand()` in C: seed it once, then call repeatedly to produce a stream of data.

`visualizer.py` is an interactive Pygame-based renderer that displays the generated curves in real time. It shows the patient's skill profile, derived parameters, and live statistics (time in range, mean BG, etc.) in a sidebar, with the main chart area rendering whichever curves are toggled on. Mouse hover shows exact values at any time point.

**Performance:** Curve contributions are pre-accumulated into numpy arrays (`_carb_totals`, `_basal_totals`, `_bolus_totals`, `_exercise_totals`) so each time step reads values in O(1). Insulin-on-board (IOB) is computed as a single `np.sum` over the future insulin array. This makes `generate_hours()` fast enough for bulk training-data generation.


## Blood Sugar Computation

At each 5-minute time step, the BG delta is computed as:

```
glucose_in  = carbs + hepatic_output - exercise
glucose_out = insulin_units * ICR / insulin_sensitivity
delta_BG    = alpha * (glucose_in - glucose_out)
```

Where `alpha` is `BG_SCALE_FACTOR`, the master scaling constant that converts abstract units to mg/dL. Insulin sensitivity divides the insulin-clearance term: resistant patients (IS > 1) clear less glucose per unit insulin, sensitive patients (IS < 1) clear more. HGO suppression by insulin is handled separately (via a Hill function on smoothed plasma insulin), so IS modulates only peripheral insulin action.

After computing the delta, three physiological guardrails are applied:

- Renal clearance: above 180 mg/dL, the kidneys excrete glucose proportionally to the excess.
- Counter-regulatory response: below 70 mg/dL, glucagon and cortisol force the liver to dump extra sugar.
- Severe-hypo glucagon dump: below `SEVERE_HYPO_THRESHOLD`, an additional emergency release adds glucose proportionally to severity.

Soft delta-damping near the floor and ceiling shapes the tails; a hard clamp at 40-400 mg/dL acts as a backstop.

Alcohol additionally suppresses HGO (on top of insulin's own suppression) by 30–70% for 4–8 hours starting 1–2 hours after drinking. Stress events temporarily multiply `insulin_sensitivity` (which is interpreted as a resistance multiplier here — higher = more resistant) by 1.2–1.5.


## Patient Model

Each virtual patient is defined by four skill dimensions sampled from a multivariate normal with configurable correlation (default 0.7):

- Dietary discipline (s1): Controls carb amounts per meal, number of meals/snacks, fast-vs-slow carb mixture, and meal timing regularity. Low s1 patients eat more fast carbs and display more erratic eating patterns.

- Attentiveness (s2): Controls how often the patient checks their CGM, how quickly they respond to highs and lows, and whether they notice overnight alarms. Also drives trend-based anticipatory corrections.

- Dosing competence (s3): Controls accuracy of carb counting, correctness of bolus timing (pre-bolus vs post-bolus), IOB awareness (high-s3 patients account for active insulin before correcting), and appropriateness of correction doses. Also controls the probability of rage eating and rage bolusing.

- Lifestyle consistency (s4): Controls regularity of wake/sleep times, exercise frequency, meal schedule stability, alcohol consumption frequency, and overall routine predictability.

These skills are mapped through a sigmoid and clipped to a configurable range (default 0.15-0.98). From these four numbers, all behavioral parameters are derived: meal sizes, timing jitter, bolus accuracy, correction behavior, exercise habits, and more.

Beyond skill, each patient also carries orthogonal physiological traits sampled independently of the skill profile: body weight, a per-patient insulin-resistance level (which sets baseline insulin needs and carb ratio and shifts the patient's average glucose — more-resistant patients run higher), and a per-patient glucose-variability scale (how widely their glucose wanders within the day). Because these vary from patient to patient, the population shows realistic between-patient spread in both mean glucose and glycemic variability, comparable to the spread seen across the real CGM cohorts.


## Insulin Sensitivity Model

Insulin sensitivity follows a multi-peak diurnal pattern modeled as a sum of Gaussian bumps:

- Morning peak (dawn phenomenon): Resistance rises around 7 AM, causing the classic morning BG rise.
- Evening rebound: A latent evening resistance peak near 8 PM, currently disabled (`IS_EVENING_AMPLITUDE = 0.0`).
- Nighttime dip: Sensitivity increases around 2 AM, which can cause nocturnal lows.

The morning peak's timing shifts randomly day-to-day (configurable sigma). A daily drift and per-step noise add further variability. During illness, the IS factor ramps gradually toward a target and ramps back down during recovery.

Additional IS modifiers apply on top of the diurnal pattern:

- **Post-exercise sensitivity boost**: After aerobic exercise, IS is reduced by `EXERCISE_IS_REDUCTION` (10%) for `EXERCISE_IS_DURATION_HOURS` (6h), modelling the well-known glucose-lowering effect of exercise that causes nocturnal hypos in active patients.
- **Stress resistance**: Stress events (2–6h duration, 1.2–1.5× IS multiplier) model the transient insulin resistance from cortisol and adrenaline.
- **Glucotoxicity**: A slow 3h EMA of true BG drives transient insulin resistance when chronically elevated, closing a positive feedback loop on hyperglycemia (high BG → more IR → harder to bring down).
- **Postprandial insulin resistance**: While carbs are absorbing, the insulin-resistance factor is multiplied by `(1 + penalty)` where `penalty` saturates with active carb load. In T1DM the incretin / GLP-1 sensitivity boost non-diabetics get with a meal is blunted/absent, so the absorbing-carb state is if anything mildly insulin-*resistant* — insulin clears glucose slightly less effectively after eating.
- **Injection site quality (lipohypertrophy)**: Every insulin dose (basal, meal bolus, corrections) is multiplied by a per-dose `site_quality` factor sampled from `N(1.0, σ)` where σ scales with `1/s4`. Patients with poor lifestyle consistency rotate sites poorly and develop higher dose-to-dose variance.


## Behavioral Events

The simulator generates the following events:

- **Meals**: Number, timing, and carb amount are all skill-dependent. Each meal is decomposed into 2-5 overlapping gamma absorption components (a "mixed meal" model): the component count is `MIXED_MEAL_MIN_COMPONENTS + Poisson(λ)` capped at the max, and carb fractions are drawn from a Dirichlet distribution. Each component is classified as fast / medium / slow with weights driven by the patient's `slow_carb_preference`, and its `(k, θ)` is uniformly sampled from category-specific ranges. A protein/fat tail is always added, sized as `PROTEIN_FAT_FRACTION_OF_CARBS × meal_carbs` and floored at `PROTEIN_FAT_MIN_GRAMS` (6 g), so snacks ~6 g, typical meals ~10–15 g, large dinners ~18 g. Hypo-correction carbs use a separate fast pair that peaks faster than meal carbs (glucose tablets / juice).

- **Basal insulin**: Long-acting insulin injected on a per-patient cadence. The total 24h-equivalent dose is anchored to `HGO_base × 24h × (body_weight_kg / BODY_WEIGHT_MEAN_KG) × is_base / ICR` — the weight factor mirrors the per-step HGO scaling and the `is_base` factor keeps the HGO-balances-basal invariant across body sizes and baseline insulin needs. Unskilled patients deviate from this ideal more. A daily adjustment mechanism lets the patient nudge their dose based on the previous day's mean BG. Each patient is assigned one of two long-acting analogues (`BASAL_VARIANTS`) — glargine (26h duration of action) or degludec (42h) — and injects once daily (`BASAL_DOSE_INTERVAL_HOURS`, 24h), with the per-injection amount equal to the 24h-equivalent dose. Absorption is modeled by a Bateman one-compartment PK curve `f(t) = exp(-ke·t) − exp(-ka·t)` (broad peak at ~6.3h, half-life ~9.9h, no plateau and no slope discontinuity) with a smootherstep tail clip. Each dose's curve is generated with duration `basal_duration_hours × (1 + BASAL_PK_OVERLAP_FRACTION)` — overlap is 1.00 so the PK lasts 2× the cadence, meaning 2–3 doses always contribute simultaneously and a single missed dose is bridged by the previous dose's still-active tail.

- **Bolus insulin**: Dosed per meal based on an estimated carb count (with skill-dependent counting error). Timing is skill-dependent: competent patients pre-bolus, incompetent ones bolus after eating. Snack boluses may be skipped. Bolus PK is dose-dependent: both duration of action and θ scale with `√dose` (centered on a 5U reference), so larger doses act longer and peak slightly later, matching observed subcutaneous insulin behavior. Use the `bolus_pk_for_dose(dose)` helper to retrieve `(k, θ, duration_minutes)`.

- **Corrections**: The patient checks their CGM at skill-dependent intervals. High-competence patients account for insulin-on-board (IOB) before correcting to avoid stacking. Attentive patients also react to BG *trends*: a rising trend above `TREND_HIGH_BG_MIN` (145 mg/dL) or a falling trend below `TREND_LOW_BG_MAX` (110 mg/dL) triggers a preemptive correction before crossing the absolute threshold. At extreme values (above 300, or below the 55 mg/dL severe-hypo threshold), rage bolusing or reflexive rescue eating may occur.

- **Exercise**: Occurs with skill-dependent probability. Modeled as a negative carb-equivalent gamma curve plus a 6h post-exercise IS sensitivity boost (`EXERCISE_IS_DURATION_HOURS`). Reduced probability on weekends.

- **Alcohol**: On weekends, holidays, and rare event days (higher probability), the patient may drink. This triggers HGO suppression (30–70%) for 4–8 hours starting 1–2 hours after drinking, causing the delayed nocturnal lows common in real T1DM patients.

- **Stress events**: Occasional transient IS increases (1.2–1.5×, 2–6h) model cortisol spikes from work, emotion, or poor sleep. Frequency decreases with lifestyle consistency.

- **Weekday/weekend/holiday patterns**: Wake time shifts later on weekends and holidays, meal timing is more variable, carb amounts are slightly larger, and alcohol probability increases. Public holidays (10–20 per year, configurable) are distributed across the year and never fall on weekends.

- **Rare events**: With low probability per day, the patient has a "chaotic day" where all skills are degraded and schedule is disrupted.

- **Illness**: With low daily probability, the patient gets sick. Illness gradually ramps up insulin resistance over several days and returns to normal during recovery.

- **Anomalous events**: With ~1% daily probability, one meal curve has its gamma shape parameters dramatically modified (k and theta multiplied by random factors), modelling bimodal absorption, injection site issues, or unexplained BG spikes.


## Installation and Usage

Requirements: Python 3.10+, numpy, pygame, pytest (for tests).

```bash
pip install numpy pygame pytest
```

Interactive visualizer:

```bash
python visualizer.py
python visualizer.py --seed 7 --bg 150 --hours 48
```

Programmatic usage:

```python
from simulator import T1DMSimulator

sim = T1DMSimulator(seed=42, initial_bg=120)

# Step-by-step generation
step = sim.generate()   # returns dict with all values for this 5-min step
step = sim.generate()   # next step

# Bulk generation
data = sim.generate_hours(72)  # returns dict of numpy arrays

# Patient info
print(sim.get_patient_summary())

# Reseed
sim.reseed(seed=99)

# Inject a curve externally (e.g., for testing or custom scenarios)
import numpy as np
from simulator import gamma_curve
curve = gamma_curve(60.0, k=2.0, theta=15.0, duration_minutes=120.0)
sim.inject_curve(curve, sim.state.current_idx, 'carb', 'Custom meal')
```


## Visualizer Controls

```
SPACE       Generate next 24 hours
R           Random reseed
1-9, 0      Toggle curve visibility
A           Toggle all curves
F           Cycle text size (small / medium / large)
Left/Right  Scroll timeline
+/-         Zoom in/out
HOME/END    Jump to start/end
Mouse       Hover for values
S           Screenshot (PNG)
Q/ESC       Quit
```

Curves: (1) Blood Glucose, (2) Carb Intake, (3) Insulin (total), (4) Basal, (5) Bolus, (6) Insulin Resistance (multiplier; >1 = resistant), (7) Exercise, (8) BG Delta, (9) Hepatic Output, (0) Glucose In.

## Parameters

All parameters are uppercase constants at the top of `simulator.py`. They are grouped by category:

- Time resolution (`DT_MINUTES`, `STEPS_PER_DAY`)
- Skill distribution (`SKILL_CORRELATION`, `SKILL_VARIANCE`, `SKILL_MIN`, `SKILL_MAX`)
- Wake/sleep schedule
- Meal generation (counts, timing, carb amounts, fast/slow mixture, curve shapes)
- Insulin sensitivity (diurnal pattern, daily drift, noise, illness effects)
- Basal insulin (sigma around HGO/ICR/weight ideal, Bateman one-compartment `basal_curve` with `BASAL_KA_PER_HOUR` / `BASAL_KE_PER_HOUR` rate constants and a smootherstep tail clip, per-analogue duration of action from `BASAL_VARIANTS` (glargine 26h, degludec 42h), once-daily injection cadence (`BASAL_DOSE_INTERVAL_HOURS`), per-dose injection-site noise damped by `BASAL_SITE_QUALITY_DAMPING`, miss probability, daily adjustment)
- Bolus insulin (curve shape, timing, carb counting error)
- Correction behavior (thresholds, patience, CGM check intervals, IOB awareness, trend thresholds)
- Exercise (probability, duration, carb equivalent, delayed IS effect)
- Hepatic glucose output
- BG computation (scale factor, clamps, guardrails)
- CGM noise
- Weekday/weekend modifiers and public holiday counts
- Alcohol (probability by day type, HGO reduction, onset delay, duration)
- Stress events (probability, IS factor range, duration range)
- Anomalous events (probability, curve shape multiplier ranges)
- Rare events and rage behavior


## Comparison Against Real-World Datasets

The simulator output is compared against three non-redistributable real CGM corpora — **OhioT1DM** (6 US adults, 5-min Medtronic Enlite CGM), **ShanghaiT1DM** (12 patients / 16 records, 15-min cadence, mixed CSII + MDI), and **AZT1D** (25 US adults on Tandem t:slim X2 Control-IQ AID systems, 5-min Dexcom G6 plus full pump event log: basal rate, bolus type, correction-vs-meal split, carb size, device mode) — on distributional moments, KS / Wasserstein / JS distances, LBGI / HBGI, MAGE / CONGA / MODD / SampEn, autocorrelation across nine lags, diurnal envelopes, weekday × hour heatmaps, episode counts and durations, hypo recovery time, per-record TIR / TBR scatter, and (AZT1D only) a head-to-head insulin / carb behaviour panel. An extended-statistics section adds further two-sample distances (energy, Cramér–von Mises, Anderson–Darling, total-variation, Hellinger, histogram-overlap), cadence-fair metrics computed on a common 15-minute grid, temporal-structure measures (Poincaré SD1/SD2, spectral entropy, DFA/Hurst, ACF e-folding time, and a glycemic-band Markov transition matrix with per-band dwell times), cross-seed bootstrap confidence intervals, and a standardised gap score.

The full report — tables, figures, and methodology — lives at [`diff/README.md`](diff/README.md). All three datasets are gitignored and live under `datasets/` (subject to data-use agreements). Reproduce with:

```bash
python diff/build_report.py                          # default corpus
python diff/build_report.py --n-seeds 300 --days 70  # larger synthetic corpus
```

## Comparison Against the UVA/Padova Simulator

Where the datasets above anchor the simulator to *real* CGM, a second comparison pits it against the field's reference *in-silico* model — the FDA-accepted **UVA/Padova 2008** simulator, via the open-source [`simglucose`](https://github.com/jxx123/simglucose) ODE core, driven through a thin adapter in [`uva_padova/padova_engine.py`](uva_padova/padova_engine.py). Three complementary lenses, each its own self-contained report:

- **Identical-input replay** — [`uva_padova/README.md`](uva_padova/README.md). The exact meals, boluses, and basal a seed generates are captured and replayed verbatim into a paired UVA/Padova virtual patient, isolating how the two physiologies answer the same behaviour. Also carries a single-threaded **generation-speed** benchmark: this simulator reads pre-accumulated curves in O(1) per 5-min step, while UVA/Padova integrates a thirteen-state stiff ODE every minute.
- **Meal excursions** — [`uva_padova/EXCURSIONS.md`](uva_padova/EXCURSIONS.md). Sharing only the meal schedule and letting each engine dose for its own physiology, the post-meal excursions are compared in amplitude, time-to-peak, area, and amplitude-normalised shape.
- **Distance to real** — [`uva_padova/REALISM.md`](uva_padova/REALISM.md). Treating each simulator as a synthetic-data source, this measures how far each one's output sits from the real cohorts above, across the same metric battery, with the distance *between* the real cohorts as the yardstick for "indistinguishable from real".

The reference engine is installed without its reinforcement-learning extras (its pinned `gym` is unneeded and does not build on current Python). Reproduce with:

```bash
pip install --no-deps simglucose>=0.2.11
python uva_padova/compare_uva_padova.py     # identical-input replay + speed
python uva_padova/compare_excursions.py     # meal-excursion deviations
python uva_padova/compare_realism.py        # distance-to-real-CGM
```

## Testing

```bash
python -m pytest tests/ -v
```

The test suite (58 tests) covers:
- `tests/test_curves.py` — curve generation correctness and unit consistency
- `tests/test_patient.py` — skill ranges, basal/HGO/ICR relationship, behavioral parameters
- `tests/test_simulator.py` — reproducibility, BG bounds, meal/insulin effects, weekday/weekend/holiday, severe-hypo rescue grams, skill-scaled correction, `inject_curve` totals contract, follow-up snack effect
- `tests/test_balance.py` — basal-HGO balance, meal-bolus balance, ICR-basal proportionality

## References

The comparison report in [`diff/README.md`](diff/README.md) benchmarks the simulator against three publicly available T1D CGM datasets. Credit and citation requests for those datasets belong to their original authors.

- **OhioT1DM** — Marling, C., and Bunescu, R. *The OhioT1DM Dataset for Blood Glucose Level Prediction: Update 2020.* Proceedings of the 5th International Workshop on Knowledge Discovery in Healthcare Data (KDH @ ECAI 2020), CEUR Workshop Proceedings, vol. 2675, pp. 71–74. Distributed under a data-use agreement via Ohio University; please request access through the maintainers' instructions before redistributing.

- **ShanghaiT1DM** — Zhao, Q., Zhu, J., Shen, X., Lin, C., Zhang, Y., Liang, Y., Cao, B., Li, J., Liu, X., Rao, W., and Wang, C. *Chinese Diabetes Datasets for Data-Driven Machine Learning.* Scientific Data 10, 35 (2023). doi:10.1038/s41597-023-01940-7. The T1DM portion contains 12 patients / 16 records of paired CGM, insulin, and dietary data.

- **AZT1D** — Khamesian, S., Arefeen, A., Thompson, B. M., Grando, M. A., and Ghasemzadeh, H. *AZT1D: A Real-World Dataset for Type 1 Diabetes.* Dataset of 25 individuals with T1D on Automated Insulin Delivery (Tandem t:slim X2 Control-IQ) collected at Mayo Clinic Arizona over 6–8 weeks per patient, including CGM, basal/bolus insulin (with correction-specific amounts and bolus types), carbohydrate intake, and device-mode annotations (regular / sleep / exercise). See the accompanying manuscript (Mayo Clinic / Arizona State University, 2025) for full study design and IRB protocol (#23-003065).

The in-silico comparison in [`uva_padova/README.md`](uva_padova/README.md) benchmarks the simulator against the UVA/Padova model, run through the open-source `simglucose` engine.

- **UVA/Padova Type 1 Diabetes Simulator** — Dalla Man, C., Rizza, R. A., and Cobelli, C. *Meal Simulation Model of the Glucose–Insulin System.* IEEE Transactions on Biomedical Engineering 54(10), 1740–1749 (2007). doi:10.1109/TBME.2007.893506. Simulator update: Dalla Man, C., Micheletto, F., Lv, D., Breton, M., Kovatchev, B., and Cobelli, C. *The UVA/PADOVA Type 1 Diabetes Simulator: New Features.* Journal of Diabetes Science and Technology 8(1), 26–34 (2014). doi:10.1177/1932296813514502. The FDA-accepted 2008 version of this model is the in-silico reference used here.

- **simglucose** — Xie, J. *simglucose: A Type-1 Diabetes Simulator as a Reinforcement Learning Environment in OpenAI Gym* (2018). An open-source Python implementation of the FDA-accepted UVA/Padova (2008) model. GitHub: <https://github.com/jxx123/simglucose> — the engine driven by the comparison scripts in `uva_padova/`.

## License

Copyright 2026 0xdeadf1sh

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
