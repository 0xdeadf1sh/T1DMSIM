Task List: T1DM Simulator Bug Fixes & Refactoring

Please implement the following fixes in the codebase.
1. Fix Critical IOB (Insulin-On-Board) Calculation Bug

File: simulator.py
Problem: Basal insulin is injected under the generic 'insulin' label and added to _insulin_totals. The _check_and_correct() method calculates IOB by summing _insulin_totals[time_idx:]. This incorrectly includes 24+ hours of unabsorbed basal insulin, causing competent patients to indefinitely skip correction boluses.
Instructions:

    Split the _insulin_totals array into two separate arrays: _basal_totals and _bolus_totals.

    Update _ensure_totals_length() to resize both arrays.

    Update _add_to_totals() to handle 'basal' and 'bolus' event types separately.

    In _generate_day_events(), change the event type for basal insulin injections to 'basal', and meal/correction boluses to 'bolus'.

    In generate(), combine them for the BG math: total_insulin = float(self._basal_totals[idx]) + float(self._bolus_totals[idx]).

    In _check_and_correct(), calculate iob by summing only future bolus insulin: iob = float(np.sum(self._bolus_totals[time_idx:])).

2. Prevent Negative/Past Event Jitter Bug

File: simulator.py
Problem: In _generate_day_events(), applying negative jitter to early morning meals/events can result in an index that is less than self.state.current_idx. Because the event loop pops events <= idx, these past events are processed immediately but scatter-added into the past, losing mass.
Instructions:

    In _generate_day_events(), wherever a time index is calculated (e.g., meal_idx, bolus_idx, basal_time_idx), wrap the result in a max() call to clamp it to at least day_start_idx or self.state.current_idx.

    Example: meal_idx = max(self.state.current_idx, day_start_idx + int(meal_time * 60 / DT_MINUTES))

3. Rename "Insulin Sensitivity" to "Insulin Resistance"

File: simulator.py
Problem: The variable insulin_sensitivity acts mathematically as a multiplier for effective_carb_load (i.e., higher values = carbs spike BG faster, requiring more insulin). This is physiologically Insulin Resistance, not sensitivity.
Instructions:

    Rename the variable insulin_sensitivity to insulin_resistance_factor in generate().

    Rename the internal method _compute_insulin_sensitivity() to _compute_insulin_resistance().

    Rename the state list sensitivity_history to resistance_history.

    Ensure the output dict key in generate() remains understandable (you can map insulin_resistance_factor to a key like 'insulin_resistance').

4. Simplify Redundant CGM Noise Math

File: simulator.py -> _compute_cgm_observation()
Problem: The noise calculation contains redundant operations that cancel out.
Instructions:

    Change: noise_sigma = CGM_NOISE_FRACTION * true_bg / 100.0 * 100

    To: noise_sigma = CGM_NOISE_FRACTION * true_bg

5. Fix Off-By-One Error in Trend Rate Calculation

File: simulator.py -> _check_and_correct()
Problem: The BG trend calculates rate of change by dividing the delta by the total number of items in the window, rather than the number of elapsed intervals (N - 1).
Instructions:

    Change: trend = (window[-1] - window[0]) / TREND_CORRECTION_WINDOW_STEPS

    To: trend = (window[-1] - window[0]) / (TREND_CORRECTION_WINDOW_STEPS - 1)

6. Synchronize Documentation with Code Reality

Files: README.md, CLAUDE.md
Problem: The docs claim basal insulin uses a flat_curve over a 42-hour duration. The codebase actually uses a trapezoidal basal_curve with a 26.0 hour duration constant.
Instructions:

    In both markdown files, replace mentions of flat_curve with basal_curve (noting its trapezoidal ramp-up/ramp-down nature).

    Change the documented basal duration references from 42h to 26h to accurately reflect the BASAL_DURATION_HOURS constant in simulator.py.
