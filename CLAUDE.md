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
- HGO (hepatic glucose output) is insulin-suppressed via a Hill function on EMA-smoothed plasma insulin. At zero insulin it climbs toward HGO_UNSUPPRESSED_GRAMS_PER_HOUR; HGO_INSULIN_HALF_MAX is tuned so a typical basal level reproduces the 8.25 g/hr balanced rate (preserves basal-balances-HGO test invariant).
- Delayed-meal HGO rebound: meals above DELAYED_HGO_MEAL_THRESHOLD_GRAMS schedule a positive HGO bump 3.5-5.5h later (delayed gluconeogenesis from amino acids + cortisol). This is the mechanism behind nocturnal hyperglycemia after a big dinner. Stored in state.meal_hgo_effects, applied additively to HGO via a trapezoidal envelope.
- Hepatic glycogen reservoir (state.glycogen_grams) is a finite store. HGO drains it (via the glycogenolysis-sourced fraction); absorbed carbs refill it. When depleted, HGO scales down toward the gluconeogenesis-only floor. Refill is a "background" channel and does not subtract from BG-bound carbs.
- Glucotoxicity: a slow 6h EMA of true BG (state.glucotox_bg_ema) drives transient insulin resistance when chronically elevated. Closes a positive feedback loop on hyperglycemia.
- Postprandial insulin resistance: while carbs are absorbing, the insulin-resistance factor is multiplied by (1 + penalty) where penalty saturates with active carb load. In T1DM the incretin / GLP-1 sensitivity boost non-diabetics get with a meal is blunted/absent; the absorbing-carb state is if anything mildly insulin-resistant, so insulin clears glucose slightly *less* effectively (NOT the reverse — do not re-introduce a sensitivity bonus).
- Injection site quality: every insulin dose (basal, meal bolus, corrections) is multiplied by a per-dose factor sampled from N(1.0, σ) where σ scales with 1/s4. Models lipohypertrophy from poor rotation.
- BG delta = α * (glucose_in − glucose_out) where glucose_out = total_insulin * ICR / IS. IS modulates insulin effectiveness, NOT carb load (HGO-vs-insulin coupling is handled separately by the Hill function above).
- Bolus duration of action scales with dose: `bolus_pk_for_dose(dose) -> (k, theta, duration_minutes)`. Larger doses act longer and peak slightly later. The legacy `BOLUS_DURATION_HOURS` constant is kept for tests but new code must use the helper.
- Basal insulin uses a Bateman one-compartment PK curve (`basal_curve`): `f(t) = exp(-ke·t) − exp(-ka·t)`, with `BASAL_KA_PER_HOUR = 0.30` and `BASAL_KE_PER_HOUR = 0.07` giving a broad peak at ~6.3h post-injection and a gentle ~9.9h-half-life decline. The shape sits between glargine (peakier, ~4h) and degludec (very flat, peak ~9h) — flatter than glargine specifically so that doses scheduled at cadence = duration-of-action still produce a near-flat cumulative trace. A smootherstep window over the final `BASAL_TAIL_CLIP_HOURS` tapers the late residual to zero so consecutive doses join without a tail-step. The legacy `BASAL_RAMP_UP_HOURS` / `BASAL_RAMP_DOWN_HOURS` constants are retained as backward-compat aliases used only by tests for warmup math.
- Per-patient basal duration of action is sampled in `generate_patient` uniformly on `[BASAL_DURATION_HOURS_MIN, BASAL_DURATION_HOURS_MAX]` (18–30h). The patient also re-doses at exactly that cadence (an 18h-duration patient injects every 18h, a 30h-duration patient every 30h — often skipping a calendar day). `basal_dose` stays as the 24h-equivalent total so the HGO/ICR balance invariant is unchanged; per-injection amount is scaled by `basal_duration_hours / 24` at scheduling time. Each dose's curve is generated with duration = `basal_duration_hours * (1 + BASAL_PK_OVERLAP_FRACTION)` (overlap = 1.00, so PK lasts 2× the cadence). That long PK tail means two-to-three doses always contribute simultaneously, so the cumulative trace stays smooth across normal handoffs AND a single missed dose is bridged by the previous dose's still-active tail rather than producing a full-cadence zero gap. `state.next_basal_due_idx` tracks the next scheduled injection across days; `_generate_day_events` loops until all basals due in today's window are queued. `basal_miss_prob` is still rolled per scheduled injection (skill-dependent) but `BASAL_MISS_PROB_BASE` is set to 0.02 — real T1D MDI patients fully skip <3% of long-acting doses (most "misses" are short delays absorbed by the PK overlap). Per-basal-dose noise is tightened too: `BASAL_DOSE_COMPETENCE_NOISE = 0.05` (real intra-individual basal CV is 5-15%) and the lipohypertrophy site-quality multiplier is contracted toward 1.0 by `BASAL_SITE_QUALITY_DAMPING` (long-acting basal sites — thigh/buttock — absorb far more consistently than rapid-acting sites). Without these dampers the trace inherited bolus-grade per-dose magnitude swings.
- Exercise is modeled as negative food intake, plus a 10h post-exercise IS sensitivity boost (`EXERCISE_IS_DURATION_HOURS`).
- Illness gradually ramps insulin sensitivity via a target/ramp system.
- Physiological guardrails: renal clearance above 180 mg/dL, counter-regulatory response below 70 mg/dL, additional glucagon dump below SEVERE_HYPO_THRESHOLD.
- Weekday/weekend/holiday patterns, alcohol (additional HGO suppression on top of insulin's), and stress events (transient IS increase) add behavioral realism.
- Curve contributions are pre-accumulated into numpy arrays for O(1) per-step reads.

## Key Files

- `simulator.py` -- core simulation engine, all parameters, patient generator, BG computation
- `visualizer.py` -- Pygame interactive visualizer (forces X11 on Wayland)
- `tests/` -- pytest suite (49 tests): test_curves, test_patient, test_simulator, test_balance
- `scripts/batch_test.py` -- run multiple seeds and print TIR/mean BG summary
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
```

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
- Population averages should track the pooled real-CGM cohort (OhioT1DM + ShanghaiT1DM) within the small-sample noise of those datasets (n = 6 + 13). Don't pin to a specific TBR/TIR/TAR triple — match the *shape* of the pooled distribution (central moments, percentiles, episode counts) as reported in `reports/REPORT.md`. Mean BG ≈ 160–165 mg/dL, GMI ≈ 7.2.
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
- Basal uses basal_curve (Bateman one-compartment PK: f(t) = exp(-ke·t) − exp(-ka·t), smootherstep tail clip) with total_amount = actual_dose and per-injection duration = patient.basal_duration_hours × (1 + BASAL_PK_OVERLAP_FRACTION). Per-patient basal_duration_hours is sampled uniformly on [BASAL_DURATION_HOURS_MIN, BASAL_DURATION_HOURS_MAX] (18–30h); injections are scheduled at that same cadence. The broad ~6.3h-peak shape plus the 2× cadence PK overlap is what ensures overnight coverage and bridges single missed doses — do not replace with gamma_curve, narrower k, or a fixed 24h duration without retuning. BASAL_DURATION_HOURS (28h) is now only a population-reference constant kept for legacy tests.
- The visualizer uses an off-screen buffer to avoid flickering on Wayland. Do not remove the double-buffering logic.
- BG_SCALE_FACTOR is the most sensitive parameter. Small changes have large effects.
- HGO_INSULIN_HALF_MAX must stay tuned so that compute_hgo_rate(typical_basal_per_step) ≈ HGO_BASE_GRAMS_PER_HOUR. Otherwise `test_hgo_basal_balance` will fail and the patient population will systematically run high or low. Verify with `compute_hgo_rate(0.086) ≈ 8.25` (the typical basal per-step under ICR_MEAN=8, HGO_BASE=8.25). Closed-form constraint: h = B*(B-6) / (96*(18-B)).
- BG bounds are enforced softly (delta-damping near floor/ceiling) followed by a hard clamp as backstop. Do not remove the hard clamp; do not raise BG_CLAMP_MIN above the soft floor — they work together.
- The _carb_totals/_basal_totals/_bolus_totals/_exercise_totals arrays are instance variables (not state), so they are reset in reseed() but not serialized. If you add serialization, include them.
