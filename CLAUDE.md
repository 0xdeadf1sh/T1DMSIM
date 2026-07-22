# T1DM Patient Behavior Simulator

## About

A seed-driven simulator that generates synthetic blood glucose data by modeling Type 1 Diabetes patient behavior (not physiology directly). The simulator produces factor curves (carb intake, insulin, insulin sensitivity, exercise) whose interactions determine blood sugar deltas. The end goal is to generate near-unlimited training data for downstream ML models that learn the relationships between patient behavior and blood sugar outcomes.

## Architecture

Single-file Python simulator (`simulator.py`) with a Pygame-based visualizer (`visualizer.py`). All simulation parameters are uppercase constants at the top of `simulator.py`. The simulator is stateful and step-based: call `generate()` to advance by 5 minutes, like `rand()` in C.

Key design decisions:
- Output is BG delta, not absolute BG. BG is accumulated from deltas.
- Patient behavior is driven by 4 correlated skill dimensions (`s1` dietary_discipline, `s2` attentiveness, `s3` dosing_competence, `s4` lifestyle_consistency) sampled from a multivariate normal.
- All curves (carb absorption, insulin action including basal) use gamma distributions with parameterized shape.
- Each meal becomes 2-5 overlapping gamma absorption components (mixed meal). Component types (fast/medium/slow) are weighted by the patient's slow_carb_preference. A protein/fat tail is always added.
- Per-patient meal size is set by a lognormal `meal_appetite` multiplier (sampled in `generate_patient`, `MEAL_APPETITE_LOG_SIGMA`), orthogonal to skill, applied to carb amount only (never meal count/timing, so the ACF-relevant meal shocks are preserved). It supplies the wide between-patient carbs/day spread (sd ~90 g) and right skew that per-meal noise cannot. `MEAL_CARB_SIGMA` is a *small* per-meal SD (16 g) — a large value truncated by the `max(0,·)` floor silently inflated the population's carb intake ~40%. `MEAL_CARB_SCALE` is the global trim that lands the pooled carbs/day on OhioT1DM (~194 g/day, median ~163). Trimming the meals to Ohio reduces meal-driven BG variance, which is why the GE/OU spread was raised to compensate (see the glucose-effectiveness bullet).
- HGO (hepatic glucose output) is insulin-suppressed via a Hill function on EMA-smoothed plasma insulin. At zero insulin it climbs toward HGO_UNSUPPRESSED_GRAMS_PER_HOUR; HGO_INSULIN_HALF_MAX is tuned so a typical basal level reproduces the 8.25 g/hr balanced rate (preserves basal-balances-HGO test invariant).
- Delayed-meal HGO rebound: meals above DELAYED_HGO_MEAL_THRESHOLD_GRAMS schedule a positive HGO bump 3.5-5.5h later (delayed gluconeogenesis from amino acids + cortisol). This is the mechanism behind nocturnal hyperglycemia after a big dinner. Stored in state.meal_hgo_effects, applied additively to HGO via a trapezoidal envelope.
- Hepatic glycogen reservoir (state.glycogen_grams) is a finite store. HGO drains it (via the glycogenolysis-sourced fraction); absorbed carbs refill it. When depleted, HGO scales down toward the gluconeogenesis-only floor. Refill is a "background" channel and does not subtract from BG-bound carbs.
- Glucotoxicity: a slow 3h EMA of true BG (state.glucotox_bg_ema) drives transient insulin resistance when chronically elevated. Closes a positive feedback loop on hyperglycemia.
- Postprandial insulin resistance: while carbs are absorbing, the insulin-resistance factor is multiplied by (1 + penalty) where penalty saturates with active carb load. In T1DM the incretin / GLP-1 sensitivity boost non-diabetics get with a meal is blunted/absent; the absorbing-carb state is if anything mildly insulin-resistant, so insulin clears glucose slightly *less* effectively (NOT the reverse — do not re-introduce a sensitivity bonus).
- Injection site quality: every insulin dose (basal, meal bolus, corrections) is multiplied by a per-dose factor sampled from N(1.0, σ) where σ scales with 1/s4. Models lipohypertrophy from poor rotation.
- BG delta = α * (glucose_in − glucose_out) where glucose_out = total_insulin * ICR / IS. IS modulates insulin effectiveness, NOT carb load (HGO-vs-insulin coupling is handled separately by the Hill function above).
- Bolus duration of action scales with dose: `bolus_pk_for_dose(dose) -> (k, theta, duration_minutes)`. Larger doses act longer and peak slightly later. The legacy `BOLUS_DURATION_HOURS` constant is kept for tests but new code must use the helper.
- Basal insulin uses a Bateman one-compartment PK curve (`basal_curve`): `f(t) = exp(-ke·t) − exp(-ka·t)`, with `BASAL_KA_PER_HOUR = 0.30` and `BASAL_KE_PER_HOUR = 0.07` giving a broad peak at ~6.3h post-injection and a gentle ~9.9h-half-life decline. The shape sits between glargine (peakier, ~4h) and degludec (very flat, peak ~9h) — flatter than glargine specifically so that doses scheduled at the fixed 24h cadence still produce a near-flat cumulative trace. A smootherstep window over the final `BASAL_TAIL_CLIP_HOURS` tapers the late residual to zero so consecutive doses join without a tail-step. The legacy `BASAL_RAMP_UP_HOURS` / `BASAL_RAMP_DOWN_HOURS` constants are retained as backward-compat aliases used only by tests for warmup math.
- Per-patient basal PK is fixed by the long-acting analogue the patient is assigned in `generate_patient`: `basal_type` is drawn uniformly from `BASAL_VARIANTS` and `basal_duration_hours = av["action_hours"]` — a discrete 26h (glargine) or 42h (degludec), NOT a uniform draw. Injection cadence is a fixed once-daily `BASAL_DOSE_INTERVAL_HOURS` (24h), decoupled from the action duration. Each once-daily injection delivers the full 24h `basal_dose` (`per_dose_factor = basal_dose_interval_hours / 24 = 1.0`), so the HGO/ICR balance invariant is unchanged. The dose's curve is generated with duration = `basal_duration_hours * 60` — no overlap multiplier. Because action_hours exceeds the 24h cadence the previous dose's tail still overlaps the next: degludec (42h) overlaps ~1.75 doses at steady state (very flat), glargine (26h) only ~1 dose (mild end-of-day waning). That overlap supplies overnight coverage — but only degludec's long tail meaningfully bridges a skipped dose; a missed glargine dose leaves a near-full-cadence basal gap. `state.next_basal_due_idx` tracks the next scheduled injection across days; `_generate_day_events` loops until all basals due in today's window are queued. `basal_miss_prob` is still rolled per scheduled injection (skill-dependent) but `BASAL_MISS_PROB_BASE` is set to 0.02 — real T1D MDI patients fully skip <3% of long-acting doses (most "misses" are short delays absorbed by the PK overlap). Per-basal-dose noise is tightened too: `BASAL_DOSE_COMPETENCE_NOISE = 0.1` (real intra-individual basal CV is 5-15%) and the lipohypertrophy site-quality multiplier is contracted toward 1.0 by `BASAL_SITE_QUALITY_DAMPING` (long-acting basal sites — thigh/buttock — absorb far more consistently than rapid-acting sites). Without these dampers the trace inherited bolus-grade per-dose magnitude swings. `BASAL_DURATION_HOURS_MIN`/`MAX` (15/34h) and `BASAL_PK_OVERLAP_FRACTION` are legacy constants no longer used in computation.
- Exercise is modeled as negative food intake, plus a 6h post-exercise IS sensitivity boost (`EXERCISE_IS_DURATION_HOURS`).
- Illness gradually ramps insulin sensitivity via a target/ramp system.
- Physiological guardrails: renal clearance above 180 mg/dL, counter-regulatory response below 70 mg/dL, additional glucagon dump below SEVERE_HYPO_THRESHOLD.
- Glucose effectiveness (Bergman minimal-model Sg): an always-on insulin-independent restoring pull `bg_delta += glucose_effectiveness * (E − bg)` toward a *stochastic* equilibrium `E(t)`. `E` is an Ornstein–Uhlenbeck process — each step it mean-reverts (timescale `GE_EQ_TAU_HOURS`) toward `ge_anchor + ge_dawn_amplitude * ge_diurnal_profile(hour)` (the dawn-phenomenon lift that superseded the legacy flat `GE_EQ_DAY_BOOST`), is perturbed by Gaussian noise of stationary std `GE_EQ_SIGMA * ge_sigma_mult` (per-patient-scaled — see the heterogeneity note below), then floored at `GE_EQ_FLOOR` (kept above the 55 mg/dL severe threshold). This is the mean-reversion the renal/counter-regulatory guardrails do NOT supply inside the 70–180 band; without it within-band BG is an under-damped integrator of net flux whose 8h autocorrelation decays far too slowly (~0.3 vs ~0 in real CGM). The strong, fast Sg pull gives BG a short correlation time (low 8h ACF) while `E`'s wandering supplies distributional spread that decorrelates within hours — decoupling spread from the ACF, which a *fixed* setpoint could achieve only by homogenizing the distribution. Per-patient Sg is sampled lognormally around `GE_RATE` (real Sg varies ~2–3× across individuals). Two axes of *between-patient* heterogeneity ride on the equilibrium (without them the OU homogenizes patients — strong Sg pull + a global sigma + floor-compression of low anchors — leaving the sim the narrowest of the four cohorts in inter-patient spread): (a) the per-patient anchor `ge_anchor` is drawn `normal(GE_EQ_ANCHOR_MEAN + GE_ANCHOR_IR_COUPLING*(ir−1), GE_EQ_ANCHOR_SIGMA)` so more insulin-resistant patients run higher — this carries the between-patient *mean*-BG spread on the high (floor-immune, hypo-safe) side, since the anchor sigma alone saturates (~14 mg/dL sd) because `GE_EQ_FLOOR` compresses low-anchor patients upward, so the widened `IR_LOGNORMAL_SIGMA` plus the strong `GE_ANCHOR_IR_COUPLING` supply the rest; (b) the per-patient `ge_sigma_mult` (lognormal, sigma `GE_SIGMA_REL_SIGMA`, clipped to `GE_SIGMA_MULT_CLIP`) scales `GE_EQ_SIGMA` in the OU step so patients differ in *within*-patient variability (a global `GE_EQ_SIGMA` made every patient equally variable). Together they lift the between-patient mean-BG sd (~13.6 → ~16.1) onto OhioT1DM/AZT1D (~16) and land the between-to-within variance ratio (~0.27) on Ohio (~0.28), leaving the pooled distribution (mean ~160, std ~61, KS ≈ 0.02) and the zero >2h severe-hypo tail intact. `GE_EQ_FLOOR` is load-bearing: because the restoring target is never *severely* hypoglycemic, the Sg pull is always *upward* in a severe low (it aids, never opposes, the severe-hypo rescue — this is what keeps the >2h severe tail at zero), and it anchors the pooled mean and low tail onto the real cohorts. `GE_EQ_SIGMA`, `GE_EQ_FLOOR`, and `GE_EQ_ANCHOR_MEAN` are co-tuned with the meal carb load (see `MEAL_CARB_SCALE` / `MEAL_APPETITE_LOG_SIGMA`): the OU supplies whatever BG variance the Ohio-sized meals do not, so trimming the meals to Ohio required raising `GE_EQ_SIGMA` (more OU variance) and lowering `GE_EQ_FLOOR` (a large sigma's floor-clipping otherwise inflates the pooled mean). Together they land the pooled BG on OhioT1DM (mean/std/GMI/J-index/M-value/TIR/TAR and every percentile within noise; pooled KS ≈ 0.02). The two behavioural dose-balance tests disable Sg (`GE_RATE = GE_RATE_MIN = 0`) via the `isolated_biology` fixture since the always-on pull would otherwise drag BG off the exact-dose-match assumption.
- Weekday/weekend/holiday patterns, alcohol (additional HGO suppression on top of insulin's), and stress events (transient IS increase) add behavioral realism.
- Curve contributions are pre-accumulated into numpy arrays for O(1) per-step reads.

## Key Files

- `simulator.py` -- core simulation engine, all parameters, patient generator, BG computation
- `visualizer.py` -- Pygame interactive visualizer (forces X11 on Wayland)
- `cache_simulator.py` -- pre-generate a compressed (blosc2 `.b2nd`) pool of trajectories to disk and emit `DATASET.md`. Direct-memmap multiprocessing fan-out, per-row seed rejection sampling. Discards any window touching the CGM clamp rails (bg_observed ≥ 399 or ≤ 41); `--hypo-oversample`/`--hypo-min-frac`/`--hypo-threshold` bias a fraction of rows toward hypoglycemia. During the transcode pass it also pools bg_observed (1 mg/dL histogram + power sums + Kovatchev LBGI/HBGI numerators, no extra I/O) and emits a `DATASET.md` "Distribution vs the baseline simulator" table comparing the cache against the unbiased-sim baseline in `diff/stats.json` (`datasets.Sim`; `--baseline-stats`/`--no-baseline`). The same transcode pass folds float64 power sums of the T1DMAI forward transforms (bg in Kovatchev risk space, carb/insulin in log1p) and emits a `normalization_stats.json` — the 3-channel `{bg_absolute, carb_intake, insulin_combined}: {mean, std}` (sample std) contract the T1DMAI model consumes — written durably before the `meta.json` sentinel. Needs `blosc2`.
- `tests/` -- pytest suite (78 tests): test_curves, test_patient, test_simulator, test_balance, test_hypo_oversample, test_norm_stats
- `scripts/batch_test.py` -- run multiple seeds and print TIR/mean BG summary
- `scripts/compare_all_datasets.py` -- dataset loaders + grid regularisation for the three real cohorts (OhioT1DM, ShanghaiT1DM, AZT1D); reused by the report engine
- `diff/build_report.py` -- comprehensive sim-vs-real statistical comparison; regenerates `diff/README.md`, `diff/stats.json`, `diff/figures/*.png`. Runs the sim for N seeds and compares against the three real cohorts (moments, percentiles, KS/Wasserstein/JS, risk indices, MAGE/CONGA/MODD, SampEn, ACF, episodes, excursions, plus the §12 extended stats). CLI: `--n-seeds N --days D --warmup-h H`
- `diff/extended_stats.py` -- pure functions for the §12 extended metrics: extra two-sample distances (energy, Cramér–von Mises at a common per-arm n, Anderson–Darling, TV/Hellinger/overlap), cadence-fair common-15-min-grid recompute, temporal structure (Poincaré SD1/SD2, Welch spectral entropy, chunked-DFA α, ACF e-folding, band-transition Markov matrix + dwell), cross-seed bootstrap CIs, and the standardised (sim − mean_real)/sd_between_real gap score
- `docs/math.md` -- mathematical formulation reference

## Commands

```bash
# Run the visualizer
python visualizer.py
python visualizer.py --seed 7 --bg 150 --hours 48

# Run the simulator standalone (quick test)
python simulator.py

# Run tests
python -m pytest tests/ -v

# Run a batch of seeds to check BG distributions
python scripts/batch_test.py

# Regenerate the full sim-vs-real statistical comparison (report + stats + figures)
python diff/build_report.py                      # default 100 seeds x 70 d
python diff/build_report.py --n-seeds 300 --days 70   # larger synthetic corpus

# Cache a compressed trajectory pool + regenerate DATASET.md (with the
# distribution-vs-baseline comparison against diff/stats.json)
python cache_simulator.py --out-dir simulator_cache --pool-size 50000
python cache_simulator.py --pool-size 50000 --hypo-oversample 0.25   # tail oversampling
```

The three real datasets must live under `datasets/` (gitignored). `diff/stats.json`
and `diff/README.md` are regenerated artefacts; the report prose is templated and
kept neutral/observational, so a re-run after simulator changes needs no hand-editing.

Visualizer key bindings are documented in the module docstring at the top of `visualizer.py` (and in the README's "Visualizer Controls" section).

## Code Style

- Python 3.10+, numpy for numerics
- Parameters as module-level uppercase constants with comments
- Type hints on all function signatures
- Dataclasses for structured data (PatientProfile, SimulatorState, ActiveCurve)
- No external dependencies beyond numpy and pygame

## Important Conventions

- The gamma_curve function normalizes so sum(values) = total_amount (values are in amount-per-step units). The basal_curve function (Bateman one-compartment PK, smootherstep tail clip) likewise normalizes so sum(values) = total dose. All curves are in amount-per-step units — do not pass rates.
- When changing the BG delta formula, always trace through the math with concrete numbers to verify the magnitudes make sense. A typical meal should produce a post-meal BG rise of 30-80 mg/dL over 1-2 hours.
- The seed determines everything. Same seed = same patient = same simulation. Always verify reproducibility after changes.
- Never add dependencies beyond numpy and pygame without discussion.
- Keep the parameter count high and the parameter names descriptive. The user wants many knobs to turn.
- Curve contributions are scatter-added into _carb_totals / _basal_totals / _bolus_totals / _exercise_totals on activation. Use inject_curve() (not state.active_curves.append) whenever inserting curves from outside generate().

## Testing Approach

- Use multiple seeds (0-20) over 72-hour runs to verify BG distributions
- Check TIR (70-180), TBR (<70), TAR (>180), severe-low (<55), mean BG across seeds
- Population averages should track OhioT1DM, the calibration target, within the small-sample noise of that dataset (n = 6). Don't pin to a specific TBR/TIR/TAR triple — match the *shape* of its distribution (central moments, percentiles, episode counts) as reported in `diff/README.md`. Mean BG ≈ 160–165 mg/dL, GMI ≈ 7.2.
- **Hypo episode shape (critical clinical invariant — non-negotiable):**
  - Severe-hypo (<55) episodes: max duration ≤ 1h, mean ≤ 0.15h, **zero** > 2h
  - Mild hypo (55-70) episodes: median ~0.75h, p90 ~1.4h, max ~6-7h (rare outliers)
  - Population time below 55: mean ≤ 2%, max patient ≤ 6%
- Skill should remain positively correlated with TIR (Skill-TIR ≈ +0.4 to +0.6) and roughly uncorrelated with TBR. Use this as a regression canary; do not target specific high/mid/low TIR percentages.
- Three structural rules to preserve:
  - `ideal_basal = HGO_BASE * 24 * (body_weight_kg / BODY_WEIGHT_MEAN_KG) * is_base / ICR` (in `generate_patient`) — the weight factor mirrors the per-step HGO scaling and keeps the HGO-balances-basal invariant across body sizes.
  - Hypo correction grams scale with `skill_avg` so skilled patients can recover from over-bolus crashes
  - Severe hypo (`bg_observed < SEVERE_HYPO_THRESHOLD`) bypasses the CGM check interval AND triggers a non-probabilistic ≥14g rescue (grows with deficit: `14 + 0.35 * deficit`) — this is what keeps severe episodes under 1h. The rescue is still gated by a short `SEVERE_HYPO_REFRACTORY_MIN` (10 min) between back-to-back doses so that stacked carbs don't sawtooth into post-correction hypers. Skill-gated slow-carb follow-up snack (`HYPO_FOLLOWUP_*`) supplies a tail so the patient doesn't immediately re-hypo. Removing any of the three pieces (severe bypass, ≥14g rescue, follow-up tail) re-opens 6+ hour dangerous hypos or 10-15min sawtooth.

## Warnings

- Both gamma_curve and basal_curve produce values in "amount per step" — never accidentally pass a rate (rate_per_hour) where total_amount is expected. Verify per-step magnitudes when changing curve generation.
- Basal uses basal_curve (Bateman one-compartment PK: f(t) = exp(-ke·t) − exp(-ka·t), smootherstep tail clip) with total_amount = actual_dose and per-injection PK duration = the assigned analogue's action_hours (glargine 26h / degludec 42h from BASAL_VARIANTS). Per-patient basal_duration_hours is set to that action_hours (NOT sampled from [15,34h]); injection cadence is a fixed once-daily BASAL_DOSE_INTERVAL_HOURS (24h), decoupled from the action duration. The long action tail (steady-state overlap ≈1.08× for glargine, ≈1.75× for degludec; tmax ≈6.8h / ≈11.5h) is what ensures overnight coverage — degludec's tail bridges a single missed dose, glargine's does not — do not replace with gamma_curve, narrower k, or a shorter fixed duration without retuning. BASAL_DURATION_HOURS (28h), BASAL_DURATION_HOURS_MIN/MAX (15/34h), and BASAL_PK_OVERLAP_FRACTION are now unused-in-computation legacy/reference constants.
- The visualizer uses an off-screen buffer to avoid flickering on Wayland. Do not remove the double-buffering logic.
- BG_SCALE_FACTOR is the most sensitive parameter. Small changes have large effects.
- HGO_INSULIN_HALF_MAX must stay tuned so that compute_hgo_rate(typical_basal_per_step) ≈ HGO_BASE_GRAMS_PER_HOUR. Otherwise `test_hgo_basal_balance` will fail and the patient population will systematically run high or low. Verify with `compute_hgo_rate(0.086) ≈ 8.25` (the typical basal per-step under ICR_MEAN=8, HGO_BASE=8.25). Closed-form constraint: h = B*(B-6) / (96*(18-B)).
- BG bounds are enforced softly (delta-damping near floor/ceiling) followed by a hard clamp as backstop. Do not remove the hard clamp; do not raise BG_CLAMP_MIN above the soft floor — they work together.
- The _carb_totals/_basal_totals/_bolus_totals/_exercise_totals arrays are instance variables (not state), so they are reset in reseed() but not serialized. If you add serialization, include them.
