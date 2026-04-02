# T1DM Patient Behavior Simulator

A seed-driven simulator for generating synthetic Type 1 Diabetes blood glucose data. Unlike traditional glucose-insulin simulators (e.g., UVA/Padova), this simulator models patient *behavior* as the primary driver of blood sugar outcomes. The simulator generates factor curves -- carbohydrate intake, insulin action, insulin sensitivity, and exercise -- and computes blood sugar as the emergent result of their interactions.

Designed by a T1DM patient, informed by lived experience.


## Motivation

Most T1DM simulators model physiology: glucose kinetics, insulin pharmacokinetics, compartmental models. They produce accurate BG traces but require dozens of physiological parameters that are hard to measure and vary between patients.

This simulator takes a different approach. It models the *person*, not the pancreas. The key insight is that most real-world blood sugar variance comes from behavioral decisions -- what the patient eats, when they bolus, how they correct, whether they exercise -- not from subtle physiological differences. By generating diverse behavioral patterns and computing BG as a consequence, the simulator produces training data that teaches a model to predict what patients *do*, with blood sugar as the outcome.

The ultimate goal is to train a transformer model on these synthetic factor curves, then fine-tune it on real patient data for personalized blood sugar prediction.


## Design Principles

The simulator is built on several core ideas:

1. The output is blood sugar *deltas*, not absolute values. BG at each step is the accumulation of all previous deltas. This keeps the model compositional and avoids needing to model absolute physiology.

2. Every factor is a curve, not a number. Eating 40g of bread and 40g of orange juice both contribute 40g of carbs, but the absorption curves have different shapes (the juice peaks faster and falls faster). The same applies to rapid-acting vs long-acting insulin.

3. Patient behavior is driven by a latent skill profile. Four correlated skill dimensions (dietary discipline, attentiveness, dosing competence, lifestyle consistency) determine everything about how a patient lives: what they eat, when they eat, how accurately they dose, how quickly they correct, whether they exercise.

4. The liver is a constant feeding session. Hepatic glucose output (HGO) is modeled as a steady stream of "food" entering the bloodstream. Basal insulin exists to counteract this stream, and the ideal basal dose is derived directly from `(HGO_rate × 24h) / ICR`.

5. Exercise is negative food. Walking, for example, pulls glucose out of the bloodstream into muscle cells. Modeling this as a negative carb-equivalent curve is a pragmatic simplification that works well for aerobic exercise. Additionally, exercise increases insulin sensitivity for 12–24 hours afterward, modeled as a time-limited IS reduction.

6. Everything is seed-driven. A single integer seed determines the patient's personality, physiology, daily schedule, meal choices, insulin doses, exercise patterns, illness events, and random noise. Same seed, same simulation, always.


## Architecture

The simulator consists of two files:

`simulator.py` contains the core engine. All tunable parameters are defined as uppercase constants at the top of the file (approximately 100 parameters). The `T1DMSimulator` class exposes a `generate()` method that advances the simulation by one 5-minute time step and returns all factor values and the resulting BG. This is analogous to `rand()` in C: seed it once, then call repeatedly to produce a stream of data.

`visualizer.py` is an interactive Pygame-based renderer that displays the generated curves in real time. It shows the patient's skill profile, derived parameters, and live statistics (time in range, mean BG, etc.) in a sidebar, with the main chart area rendering whichever curves are toggled on. Mouse hover shows exact values at any time point.

**Performance:** Curve contributions are pre-accumulated into numpy arrays (`_carb_totals`, `_insulin_totals`, `_exercise_totals`) so each time step reads values in O(1). Insulin-on-board (IOB) is computed as a single `np.sum` over the future insulin array. This makes `generate_hours()` fast enough for bulk training-data generation.


## Blood Sugar Computation

At each 5-minute time step, the BG delta is computed as:

```
effective_carb_load = (carbs + hepatic_output - exercise) * insulin_sensitivity
insulin_carb_equiv = insulin_units * ICR
delta_BG = alpha * (effective_carb_load - insulin_carb_equiv)
```

Where `alpha` is `BG_SCALE_FACTOR`, the master scaling constant that converts abstract units to mg/dL. After computing the delta, two physiological guardrails are applied:

- Renal clearance: above 180 mg/dL, the kidneys excrete glucose proportionally to the excess.
- Counter-regulatory response: below 70 mg/dL, glucagon and cortisol force the liver to dump extra sugar.

The resulting BG is clamped to 40-500 mg/dL.

Alcohol modifies this formula by suppressing `hepatic_output` by 30–70% for 4–8 hours starting 1–2 hours after drinking. Stress events temporarily multiply `insulin_sensitivity` by 1.1–1.5.


## Patient Model

Each virtual patient is defined by four skill dimensions sampled from a multivariate normal with configurable correlation (default 0.7):

- Dietary discipline (s1): Controls carb amounts per meal, number of meals/snacks, fast-vs-slow carb mixture, and meal timing regularity. Low s1 patients eat more fast carbs and display more erratic eating patterns.

- Attentiveness (s2): Controls how often the patient checks their CGM, how quickly they respond to highs and lows, and whether they notice overnight alarms. Also drives trend-based anticipatory corrections.

- Dosing competence (s3): Controls accuracy of carb counting, correctness of bolus timing (pre-bolus vs post-bolus), IOB awareness (high-s3 patients account for active insulin before correcting), and appropriateness of correction doses. Also controls the probability of rage eating and rage bolusing.

- Lifestyle consistency (s4): Controls regularity of wake/sleep times, exercise frequency, meal schedule stability, alcohol consumption frequency, and overall routine predictability.

These skills are mapped through a sigmoid and clipped to a configurable range (default 0.55-0.95). From these four numbers, all behavioral parameters are derived: meal sizes, timing jitter, bolus accuracy, correction behavior, exercise habits, and more.


## Insulin Sensitivity Model

Insulin sensitivity follows a multi-peak diurnal pattern modeled as a sum of Gaussian bumps:

- Morning peak (dawn phenomenon): Resistance rises around 7 AM, causing the classic morning BG rise.
- Afternoon dip: Resistance decreases in the early afternoon, making BG easier to control.
- Evening rebound: Resistance rises again around 8 PM.
- Nighttime dip: Sensitivity increases around 2 AM, which can cause nocturnal lows.

The morning peak's timing shifts randomly day-to-day (configurable sigma). A daily drift and per-step noise add further variability. During illness, the IS factor ramps gradually toward a target (rather than jumping instantly) and ramps back down during recovery.

Two additional IS modifiers apply on top of the diurnal pattern:

- **Post-exercise sensitivity boost**: After aerobic exercise, IS is reduced by `EXERCISE_IS_REDUCTION` (10%) for `EXERCISE_IS_DURATION_HOURS` (18h), modelling the well-known glucose-lowering effect of exercise that causes nocturnal hypos in active patients.
- **Stress resistance**: Stress events (2–6h duration, 1.1–1.5× IS multiplier) model the transient insulin resistance from cortisol and adrenaline.


## Behavioral Events

The simulator generates the following events:

- **Meals**: Number, timing, and carb amount are all skill-dependent. Each meal generates two overlapping gamma curves — a fast-carb fraction and a slow-carb fraction — whose ratio (`fast_fraction`) is sampled per meal and influenced by dietary discipline. A protein/fat slow-carb equivalent is added to every meal.

- **Basal insulin**: Administered once daily. The ideal dose is computed as `(HGO_rate × 24h) / ICR`; unskilled patients deviate from this ideal more. A daily adjustment mechanism lets the patient nudge their dose based on the previous day's mean BG. Absorption is modeled using a trapezoidal `basal_curve` (ramp-up then ramp-down) with a total duration of `BASAL_DURATION_HOURS` (26h), which ensures constant coverage throughout the day and overnight.

- **Bolus insulin**: Dosed per meal based on an estimated carb count (with skill-dependent counting error). Timing is skill-dependent: competent patients pre-bolus, incompetent ones bolus after eating. Snack boluses may be skipped.

- **Corrections**: The patient checks their CGM at skill-dependent intervals. High-competence patients account for insulin-on-board (IOB) before correcting to avoid stacking. Attentive patients also react to BG *trends*: a rising trend above 140 mg/dL or a falling trend below 100 mg/dL triggers a preemptive correction before crossing the absolute threshold. At extreme values (above 300 or below 55), rage bolusing or rage eating may occur.

- **Exercise**: Occurs with skill-dependent probability. Modeled as a negative carb-equivalent gamma curve plus a 12–24h post-exercise IS sensitivity boost. Reduced probability on weekends.

- **Alcohol**: On weekends, holidays, and rare event days (higher probability), the patient may drink. This triggers HGO suppression (30–70%) for 4–8 hours starting 1–2 hours after drinking, causing the delayed nocturnal lows common in real T1DM patients.

- **Stress events**: Occasional transient IS increases (1.1–1.5×, 2–6h) model cortisol spikes from work, emotion, or poor sleep. Frequency decreases with lifestyle consistency.

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
1-6         Toggle curve visibility
A           Toggle all curves
Left/Right  Scroll timeline
+/-         Zoom in/out
HOME/END    Jump to start/end
Mouse       Hover for values
S           Screenshot (PNG)
Q/ESC       Quit
```

Curves: (1) Blood Glucose, (2) Carb Intake, (3) Insulin, (4) Insulin Sensitivity, (5) Exercise, (6) BG Delta.


## Parameters

All parameters are uppercase constants at the top of `simulator.py`. They are grouped by category:

- Time resolution (`DT_MINUTES`, `STEPS_PER_DAY`)
- Skill distribution (`SKILL_CORRELATION`, `SKILL_VARIANCE`, `SKILL_MIN`, `SKILL_MAX`)
- Wake/sleep schedule
- Meal generation (counts, timing, carb amounts, fast/slow mixture, curve shapes)
- Insulin sensitivity (diurnal pattern, daily drift, noise, illness effects)
- Basal insulin (sigma around HGO/ICR ideal, gamma curve shape `BASAL_GAMMA_K/THETA` with per-dose noise, duration, miss probability, daily adjustment)
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


## Testing

```bash
python -m pytest tests/ -v
```

The test suite (38 tests) covers:
- `tests/test_curves.py` — curve generation correctness and unit consistency
- `tests/test_patient.py` — skill ranges, basal/HGO/ICR relationship, behavioral parameters
- `tests/test_simulator.py` — reproducibility, BG bounds, meal/insulin effects, weekday/weekend/holiday
- `tests/test_balance.py` — basal-HGO balance, meal-bolus balance, ICR-basal proportionality
