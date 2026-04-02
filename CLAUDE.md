# T1DM Patient Behavior Simulator

## About

A seed-driven simulator that generates synthetic blood glucose data by modeling Type 1 Diabetes patient behavior (not physiology directly). The simulator produces factor curves (carb intake, insulin, insulin sensitivity, exercise) whose interactions determine blood sugar deltas. The end goal is to generate training data for a transformer model that learns the relationships between patient behavior and blood sugar outcomes.

## Architecture

Single-file Python simulator (`simulator.py`) with a Pygame-based visualizer (`visualizer.py`). All simulation parameters are uppercase constants at the top of `simulator.py`. The simulator is stateful and step-based: call `generate()` to advance by 5 minutes, like `rand()` in C.

Key design decisions:
- Output is BG delta, not absolute BG. BG is accumulated from deltas.
- Patient behavior is driven by 4 correlated skill dimensions sampled from a multivariate normal.
- All curves (carb absorption, insulin action including basal) use gamma distributions with parameterized shape.
- HGO (hepatic glucose output) models the liver as a constant "feeding session".
- Basal insulin uses a trapezoidal curve (`basal_curve`) with a ramp-up and ramp-down phase over `BASAL_DURATION_HOURS` (26h). This ensures continuous overnight coverage.
- Exercise is modeled as negative food intake, plus a 12-24h post-exercise IS sensitivity boost.
- Illness gradually ramps insulin sensitivity via a target/ramp system.
- Physiological guardrails (renal clearance, counter-regulatory response) prevent extreme BG values.
- Weekday/weekend/holiday patterns, alcohol (HGO suppression), and stress events (transient IS increase) add behavioral realism.
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

## Code Style

- Python 3.10+, numpy for numerics
- Parameters as module-level uppercase constants with comments
- Type hints on all function signatures
- Dataclasses for structured data (PatientProfile, SimulatorState, ActiveCurve)
- No external dependencies beyond numpy and pygame

## Important Conventions

- The gamma_curve function normalizes so sum(values) = total_amount (values are in amount-per-step units). The flat_curve function uses rate_per_hour * (dt / 60) which gives amount-per-step. Both must produce values in the same units.
- Basal insulin now uses gamma_curve (not flat_curve). The total dose is passed as total_amount so the normalization guarantee still holds — do not convert to a rate first.
- When changing the BG delta formula, always trace through the math with concrete numbers to verify the magnitudes make sense. A typical meal should produce a post-meal BG rise of 30-80 mg/dL over 1-2 hours.
- The seed determines everything. Same seed = same patient = same simulation. Always verify reproducibility after changes.
- Never add dependencies beyond numpy and pygame without discussion.
- Keep the parameter count high and the parameter names descriptive. The user wants many knobs to turn.
- Curve contributions are scatter-added into _carb_totals / _basal_totals / _bolus_totals / _exercise_totals on activation. Use inject_curve() (not state.active_curves.append) whenever inserting curves from outside generate().

## Testing Approach

- Use multiple seeds (0-20) over 72-hour runs to verify BG distributions
- Check TIR (time in range 70-180), mean BG, and min/max across seeds
- A good distribution: most patients TIR 40-80%, mean BG 120-200, rare extremes
- Verify that skilled patients (high skills) have higher TIR than unskilled ones

## Warnings

- The unit mismatch between gamma_curve and flat_curve has been a recurring source of bugs. Any changes to curve generation must be verified by printing per-step magnitudes.
- Basal uses basal_curve (trapezoidal) with total_amount = actual_dose and duration = BASAL_DURATION_HOURS (26h). If you switch to gamma_curve, pick k/theta carefully — high k produces a narrow peak that falls to zero well before 24h, leaving no overnight coverage.
- The visualizer uses an off-screen buffer to avoid flickering on Wayland. Do not remove the double-buffering logic.
- BG_SCALE_FACTOR is the most sensitive parameter. Small changes have large effects.
- The _carb_totals/_basal_totals/_bolus_totals/_exercise_totals arrays are instance variables (not state), so they are reset in reseed() but not serialized. If you add serialization, include them.
