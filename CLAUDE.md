# T1DM Patient Behavior Simulator

## About

A seed-driven simulator that generates synthetic blood glucose data from Type 1 Diabetes patient behavior. It produces factor curves (carb intake, insulin, insulin sensitivity, exercise) whose interactions determine blood sugar deltas, as pretraining data for downstream models.

## Architecture

Single-file Python simulator (`simulator.py`) with a Pygame visualizer (`visualizer.py`). All parameters are uppercase constants at the top of `simulator.py`. The simulator is stateful and step-based: `generate()` advances 5 minutes.

Key design decisions:
- Output is BG delta; BG accumulates from deltas.
- Four correlated skills (`s1` dietary_discipline, `s2` attentiveness, `s3` dosing_competence, `s4` lifestyle_consistency) from a multivariate normal drive meals, CGM checks, rescue behavior, basal-dose noise, exercise and injection-site quality. Bolus dosing does not depend on skill.
- Meals, boluses, rescue carbs and exercise are gamma curves; basal is a Bateman curve. Every curve sums to its event total.
- Each meal becomes 2-5 gamma components weighted by `slow_carb_preference`. The components sum to the meal's logged grams; meal events carry `meal_key` and `meal_grams`. Nothing adds carbs a meal did not log.
- `meal_appetite` is a per-patient lognormal carb multiplier, orthogonal to skill. `MEAL_CARB_SCALE` sets population meal carbs/day.
- **Bolus policy.** Count, clock time and dose are drawn independent of meals, carbs and BG. Each day has two Poisson streams: a night window (`BOLUS_NIGHT_START_HOUR`, `BOLUS_NIGHT_HOURS`) and the rest of the day. Each day's units clear that day's planned meal carbs, its delayed liver rebounds, and the liver output the basal leaves (`_daily_bolus_units`, `hgo_surplus_grams_per_day`), times `BOLUS_BALANCE_GAIN`. They split by `BOLUS_NIGHT_UNIT_SHARE`; each dose is lognormal around its window's median. Tight meal timing (`MEAL_TIME_JITTER_BASE_MIN`, `WAKE_TIME_SIGMA_BASE`) keeps the night window meal-free, so night boluses have no meal carbs nearby. The only BG-reactive dosing is the pre-bolus CGM glance: skip below `hypo_threshold`, reduce within `BOLUS_REDUCE_MARGIN` above it. There are no correction boluses and no basal titration. Do not reintroduce meal- or BG-matched dosing: the insulin channel must stay separable from carbs (report §11.4, §11.6).
- HGO is insulin-suppressed via a Hill function on EMA-smoothed insulin. Basal is sized to cancel `HGO_BASE_GRAMS_PER_HOUR`, below the Hill output at a typical basal level, so boluses carry most of the total daily dose.
- Delayed-meal HGO rebound: meals above `DELAYED_HGO_MEAL_THRESHOLD_GRAMS` add an HGO bump 3.5-5.5h later via a trapezoidal envelope (`state.meal_hgo_effects`).
- Hepatic glycogen (`state.glycogen_grams`) is drained by HGO and refilled by absorbed carbs; when low, HGO scales toward the gluconeogenesis floor. Refill does not subtract from BG-bound carbs.
- Glucotoxicity: a 3h EMA of true BG raises insulin resistance when chronically high.
- Postprandial insulin resistance: absorbing carbs multiply the insulin-resistance factor by (1 + penalty), saturating in carb load. Never a sensitivity bonus.
- Injection site quality: every dose is multiplied by `N(1, σ)`, σ scaling with 1/s4; basal sites are damped by `BASAL_SITE_QUALITY_DAMPING`.
- `bg_delta = BG_SCALE_FACTOR * (glucose_in − glucose_out) + Sg * (E − bg)`, with `glucose_out = insulin * icr / IS`. Absorption noise perturbs the carbs and insulin that reach the blood; the recorded `total_carb`, `rescue_carb`, `basal_insulin` and `bolus_insulin` channels stay the declared curves.
- Physiology and sensor noise draw from `noise_rng`; behavior draws from `rng`. A behavior draw never shifts physiology noise, so forks of one simulator share it.
- Guardrails: renal clearance above `RENAL_THRESHOLD` runs at the UVA/Padova rate; counter-regulation below 70 and the glucagon term below `SEVERE_HYPO_THRESHOLD` are weak, so per-unit insulin response is flat across starting BG. True BG has no floor. A soft ceiling (headroom measured from `max(bg, BG_SOFT_CEILING)`, monotone in dose) and a hard clamp at `BG_CLAMP_MAX` bound it above. The CGM reading is clipped to `[BG_CLAMP_MIN, BG_CLAMP_MAX]`.
- Bolus duration of action scales with dose: `bolus_pk_for_dose(dose, k, theta, dia)`. Defaults are aspart's (`BOLUS_VARIANTS`).
- Basal PK comes from the assigned analogue (`BASAL_VARIANTS`: glargine 26h, degludec 42h), dosed once daily at the full `basal_dose` on `BASAL_DOSE_INTERVAL_HOURS`. Action outlasting the cadence supplies overnight overlap; only degludec bridges a missed dose. `BASAL_DURATION_HOURS_MIN`/`MAX` and `BASAL_PK_OVERLAP_FRACTION` are unused.
- One per-patient definition of low: `PatientProfile.hypo_threshold` (`HYPO_THRESHOLD_MEDIAN` ± `HYPO_THRESHOLD_SKILL_SPAN` on (s2+s3)/2). It triggers the rescue, gates every bolus, and (plus `EXERCISE_HYPO_MARGIN`) blocks exercise. `BG_LOW_THRESHOLD` is a label, not a cutoff.
- Rescue: the patient checks the CGM every `cgm_check_interval_min` while awake; asleep, only a reading below `SEVERE_HYPO_THRESHOLD` wakes them. Grams lift the projected BG to `hypo_threshold + HYPO_RESCUE_TARGET_MARGIN` (`HYPO_RESCUE_*`). The projection adds rescue carbs still absorbing (`_rescue_totals`) times `BG_SCALE_FACTOR` times a competence-scaled awareness; a rage-eat roll drops it. Refractory timers space repeat doses; each rescue opens a post-hypo basal suspend window; a falling trend below `TREND_LOW_BG_MAX` triggers small preemptive carbs.
- CGM lag is per patient (`cgm_lag_minutes`), a first-order diffusion before multiplicative AR(1) sensor noise.
- Glucose effectiveness `GE_RATE` is zero. The OU equilibrium `E(t)` it would pull toward (anchor + dawn profile, per-patient `ge_anchor` and `ge_sigma_mult`, `GE_EQ_FLOOR`) still runs and moves nothing. Below `RENAL_THRESHOLD` only insulin brings BG down. Any insulin-independent pull teaches that BG falls on its own.
- Dawn: a per-patient HGO surge at `DAWN_HGO_PEAK_HOUR` and the equilibrium's diurnal profile.
- Weekday/weekend/holiday patterns, alcohol, stress, illness and exercise (negative food plus a post-exercise sensitivity boost) add variation.
- Curves scatter-add into per-step numpy arrays for O(1) reads. `injection_log` records `(start_idx, curve_type, total, label)` for every injected curve.

## Key Files

- `simulator.py` -- engine, parameters, patient generator, BG computation
- `visualizer.py` -- Pygame visualizer (forces X11 on Wayland)
- `cache_simulator.py` -- pre-generates a blosc2 (`.b2nd`) trajectory pool and `DATASET.md`. Rows keep `--sim-hours` after a `--warmup-hours` lead-in (defaults 199.5 h kept, `SIMULATOR_WARMUP_HOURS` discarded); `T1DMAI/data.py`'s `ON_THE_FLY_SIM_HOURS` must match. `--hypo-oversample`/`--hypo-min-frac`/`--hypo-threshold` bias rows toward hypoglycemia. The transcode pass writes the distribution-vs-baseline table (against `diff/stats.json`) and `normalization_stats.json`, the 4-channel `{mean, std}` contract T1DMAI consumes, before the `meta.json` sentinel. A non-positive std aborts the build. Needs `blosc2`.
- `tests/` -- pytest suite: test_curves, test_patient, test_simulator, test_balance, test_hypo_oversample, test_norm_stats, test_cache_compat_meta (the last three need `blosc2`)
- `scripts/batch_test.py` -- multi-seed TIR/mean BG summary
- `scripts/compare_all_datasets.py` -- loaders and grid regularisation for OhioT1DM, ShanghaiT1DM, AZT1D
- `diff/build_report.py` -- sim-vs-real comparison; regenerates `diff/README.md`, `diff/stats.json`, `diff/figures/*.png`. CLI: `--n-seeds N --days D --warmup-h H --probe-seeds P`
- `diff/extended_stats.py` -- §10 metrics: extra distances, cadence-fair recompute, temporal structure, bootstrap CIs, gap score
- `diff/dynamics_stats.py` -- §11 simulator-only dynamics: open-loop dose-response and monotonicity on paired forks, per-unit response by starting BG, closed-loop retention, bolus context and isolated boluses, rescue per episode, carb-channel balance, meal–bolus mutual information
- `uva_padova/compare_uva_padova.py`, `compare_realism.py`, `compare_excursions.py` -- comparisons against UVA/Padova (simglucose). Their `README.md`, `REALISM.md`, `EXCURSIONS.md` are generated; regenerate, never hand-edit
- `docs/math.md` -- mathematical formulation reference

## Commands

```bash
python visualizer.py
python visualizer.py --seed 7 --bg 150 --hours 48
python simulator.py
python -m pytest tests/ -v
python scripts/batch_test.py
python diff/build_report.py                              # 100 seeds x 70 d
python diff/build_report.py --n-seeds 300 --days 70 --probe-seeds 50
python cache_simulator.py --out-dir simulator_cache --pool-size 50000
python cache_simulator.py --pool-size 50000 --hypo-oversample 0.25
```

The three real datasets live under `datasets/` (gitignored). `diff/stats.json` and `diff/README.md` are regenerated artefacts with templated, observational prose.

Visualizer key bindings are in the README's "Visualizer Controls" section.

## Code Style

- Python 3.10+, numpy for numerics
- Parameters as module-level uppercase constants with comments
- Type hints on all function signatures
- Dataclasses for structured data (PatientProfile, SimulatorState, ActiveCurve)
- No external dependencies beyond numpy and pygame

## Important Conventions

- `gamma_curve` and `basal_curve` return amount per step, summing to the total. Each gamma element integrates the density across its step (`GAMMA_CURVE_SUBSTEPS` midpoints from t=0). Never pass a rate.
- When changing the BG delta formula, trace the magnitudes with concrete numbers.
- The seed determines everything. Verify reproducibility after changes.
- Keep the parameter count high and names descriptive.
- Use `inject_curve()` to add curves from outside `generate()`. `correction_carb` curves also shadow into `_rescue_totals`, which the rescue projection and the `rescue_carb` channel read.

## Testing Approach

- The dynamics gates are report §11 (`diff/dynamics_stats.py`): open-loop dose-response non-increasing in dose, per-unit response flat across starting BG and across dose, isolated boluses followed by falling BG, meal–bolus mutual information under its week-shift null, rescue size per episode, meal carb channel equal to logged grams.
- Probes fork one simulator; open-loop arms freeze all behavior (`dynamics_stats.freeze`), including the rescue path and basal suspend.
- `ideal_basal = HGO_BASE_GRAMS_PER_HOUR * 24 * (body_weight_kg / BODY_WEIGHT_MEAN_KG) * is_base / icr` in `generate_patient`; the weight factor mirrors per-step HGO scaling.
- Without the rescue projection a deep low cascades into a dose every refractory interval.

## Warnings

- Basal uses `basal_curve` with the analogue's action hours; do not replace with `gamma_curve` or a shorter duration.
- The visualizer's off-screen buffer prevents flicker on Wayland. Keep it.
- `BG_SCALE_FACTOR` is the most sensitive parameter.
- Do not raise `GE_RATE` or `RENAL_CLEARANCE_RATE` to shorten episodes. Glucose leaving without insulin is exactly what the insulin channel must not be confounded with.
- True BG can go below zero. Only `bg_observed` is clipped; log transforms and the cache read `bg_observed`.
- `BG_INITIAL_FLOOR` (40) floors a run's initial BG draw.
- `_carb_totals`/`_basal_totals`/`_bolus_totals`/`_exercise_totals`/`_rescue_totals` and `injection_log` are instance variables reset in `reseed()`, not serialized.
