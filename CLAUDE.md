# T1DM Patient Behavior Simulator

## About

A seed-driven simulator that generates synthetic blood glucose data by modeling Type 1 Diabetes patient behavior (not physiology directly). The simulator produces factor curves (carb intake, insulin, insulin sensitivity, exercise) whose interactions determine blood sugar deltas. The end goal is to generate training data for a transformer model that learns the relationships between patient behavior and blood sugar outcomes.

## Architecture

Single-file Python simulator (`simulator.py`) with a Pygame-based visualizer (`visualizer.py`). All simulation parameters are uppercase constants at the top of `simulator.py`. The simulator is stateful and step-based: call `generate()` to advance by 5 minutes, like `rand()` in C.

Key design decisions:
- Output is BG delta, not absolute BG. BG is accumulated from deltas.
- Patient behavior is driven by 4 correlated skill dimensions (`s1` dietary_discipline, `s2` attentiveness, `s3` dosing_competence, `s4` lifestyle_consistency) sampled from a multivariate normal.
- All curves (carb absorption, insulin action including basal) use gamma distributions with parameterized shape.
- Each meal becomes 2-5 overlapping gamma absorption components (mixed meal). Component types (fast/medium/slow) are weighted by the patient's slow_carb_preference. A protein/fat tail is always added.
- HGO (hepatic glucose output) is insulin-suppressed via a Hill function on EMA-smoothed plasma insulin. At zero insulin it climbs toward HGO_UNSUPPRESSED_GRAMS_PER_HOUR; HGO_INSULIN_HALF_MAX is tuned so a typical basal level reproduces the legacy ~9 g/hr balanced rate (preserves basal-balances-HGO test invariant).
- Delayed-meal HGO rebound: meals above DELAYED_HGO_MEAL_THRESHOLD_GRAMS schedule a positive HGO bump 3.5-5.5h later (delayed gluconeogenesis from amino acids + cortisol). This is the mechanism behind nocturnal hyperglycemia after a big dinner. Stored in state.meal_hgo_effects, applied additively to HGO via a trapezoidal envelope.
- Hepatic glycogen reservoir (state.glycogen_grams) is a finite store. HGO drains it (via the glycogenolysis-sourced fraction); absorbed carbs refill it. When depleted, HGO scales down toward the gluconeogenesis-only floor. Refill is a "background" channel and does not subtract from BG-bound carbs.
- Glucotoxicity: a slow 6h EMA of true BG (state.glucotox_bg_ema) drives transient insulin resistance when chronically elevated. Closes a positive feedback loop on hyperglycemia.
- Postprandial IS bonus: while carbs are absorbing, IS is multiplied by (1 - bonus) where bonus saturates with active carb load. Models incretin / GLP-1 effect.
- Injection site quality: every insulin dose (basal, meal bolus, corrections) is multiplied by a per-dose factor sampled from N(1.0, σ) where σ scales with 1/s4. Models lipohypertrophy from poor rotation.
- BG delta = α * (glucose_in − glucose_out) where glucose_out = total_insulin * ICR / IS. IS modulates insulin effectiveness, NOT carb load (HGO-vs-insulin coupling is handled separately by the Hill function above).
- Bolus duration of action scales with dose: `bolus_pk_for_dose(dose) -> (k, theta, duration_minutes)`. Larger doses act longer and peak slightly later. The legacy `BOLUS_DURATION_HOURS` constant is kept for tests but new code must use the helper.
- Basal insulin uses a trapezoidal curve (`basal_curve`) with a ramp-up and ramp-down phase over `BASAL_DURATION_HOURS` (28h). This ensures continuous overnight coverage.
- Exercise is modeled as negative food intake, plus an 18h post-exercise IS sensitivity boost (`EXERCISE_IS_DURATION_HOURS`).
- Illness gradually ramps insulin sensitivity via a target/ramp system.
- Physiological guardrails: renal clearance above 180 mg/dL, counter-regulatory response below 70 mg/dL, additional glucagon dump below SEVERE_HYPO_THRESHOLD.
- Weekday/weekend/holiday patterns, alcohol (additional HGO suppression on top of insulin's), and stress events (transient IS increase) add behavioral realism.
- Curve contributions are pre-accumulated into numpy arrays for O(1) per-step reads.

## Key Files

- `simulator.py` -- core simulation engine, all parameters, patient generator, BG computation
- `visualizer.py` -- Pygame interactive visualizer (forces X11 on Wayland)
- `tests/` -- pytest suite (38 tests): test_curves, test_patient, test_simulator, test_balance
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

- The gamma_curve function normalizes so sum(values) = total_amount (values are in amount-per-step units). The basal_curve function (trapezoidal) likewise normalizes so the area equals the total dose. All curves are in amount-per-step units — do not pass rates.
- When changing the BG delta formula, always trace through the math with concrete numbers to verify the magnitudes make sense. A typical meal should produce a post-meal BG rise of 30-80 mg/dL over 1-2 hours.
- The seed determines everything. Same seed = same patient = same simulation. Always verify reproducibility after changes.
- Never add dependencies beyond numpy and pygame without discussion.
- Keep the parameter count high and the parameter names descriptive. The user wants many knobs to turn.
- Curve contributions are scatter-added into _carb_totals / _basal_totals / _bolus_totals / _exercise_totals on activation. Use inject_curve() (not state.active_curves.append) whenever inserting curves from outside generate().

## Testing Approach

- Use multiple seeds (0-20) over 72-hour runs to verify BG distributions
- Check TIR (70-180), TBR (<70), TAR (>180), severe-low (<55), mean BG across seeds
- Target population averages (n≈100, 96h each): **TBR ~10, TIR ~53, TAR ~37**. Mean BG ~150
- Hypo episode shape (critical clinical invariant):
  - Severe-hypo (<55) episodes: max duration ≤ 1h, mean ≤ 0.15h, **zero** > 2h
  - Mild hypo (55-70) episodes: median ~0.75h, p90 ~1.4h, max ~6-7h (rare outliers)
  - Population time below 55: mean ≤ 2%, max patient ≤ 6%
- Skill stratification:
  - **High-skill** (avg s > 0.6): TIR ~61%, TBR ~10%, TAR ~29%
  - **Mid-skill** (0.45-0.6): TIR ~55%, TBR ~12%, TAR ~33%
  - **Low-skill** (<0.45): TIR ~43%, TBR ~8%, TAR ~48% (hyper-dominant — missed basal, big carb errors)
- Skill-TIR correlation should be ~+0.5; Skill-TBR near zero
- Three structural rules to preserve:
  - `ideal_basal = HGO_BASE * 24 * (body_weight_kg / BODY_WEIGHT_MEAN_KG) * is_base / ICR` (in `generate_patient`) — the weight factor mirrors the per-step HGO scaling and keeps the HGO-balances-basal invariant across body sizes.
  - Hypo correction grams scale with `skill_avg` so skilled patients can recover from over-bolus crashes
  - Severe hypo (`bg_observed < SEVERE_HYPO_THRESHOLD`) bypasses the CGM check interval AND triggers a non-probabilistic 14-22g rage-eat — this is what keeps severe episodes under 1h. Removing either half re-opens 6+ hour dangerous hypos.

## Warnings

- Both gamma_curve and basal_curve produce values in "amount per step" — never accidentally pass a rate (rate_per_hour) where total_amount is expected. Verify per-step magnitudes when changing curve generation.
- Basal uses basal_curve (trapezoidal: ramp-up → plateau → ramp-down) with total_amount = actual_dose and duration = BASAL_DURATION_HOURS (28h); the plateau is what ensures overnight coverage. Do not replace this with gamma_curve without retuning — high k produces a narrow peak that falls to zero well before 24h, leaving no overnight coverage.
- The visualizer uses an off-screen buffer to avoid flickering on Wayland. Do not remove the double-buffering logic.
- BG_SCALE_FACTOR is the most sensitive parameter. Small changes have large effects.
- HGO_INSULIN_HALF_MAX must stay tuned so that compute_hgo_rate(typical_basal_per_step) ≈ HGO_BASE_GRAMS_PER_HOUR. Otherwise `test_hgo_basal_balance` will fail and the patient population will systematically run high or low. Verify with `compute_hgo_rate(0.07)` ≈ 9.
- BG bounds are enforced softly (delta-damping near floor/ceiling) followed by a hard clamp as backstop. Do not remove the hard clamp; do not raise BG_CLAMP_MIN above the soft floor — they work together.
- The _carb_totals/_basal_totals/_bolus_totals/_exercise_totals arrays are instance variables (not state), so they are reset in reseed() but not serialized. If you add serialization, include them.
