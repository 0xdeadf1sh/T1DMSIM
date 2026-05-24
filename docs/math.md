# Mathematical Formulation

Reference document for the T1DM simulator's mathematical model. Consult this when modifying the BG delta computation or curve generation.


## Patient Skill Profile

Sample from a 4D multivariate normal:

    s_raw = (s1, s2, s3, s4) ~ N(0, Sigma)

Where Sigma has `SKILL_VARIANCE` on the diagonal and `SKILL_CORRELATION * SKILL_VARIANCE` on the off-diagonals.

After sampling, apply sigmoid and clamp:

    s_i = sigmoid(s_raw_i) = 1 / (1 + exp(-s_raw_i))
    s_i = clip(s_i, SKILL_MIN, SKILL_MAX)


## Carbohydrate Absorption Curves

Each meal becomes 2-5 overlapping gamma-distributed absorption curves:

    C_i(t) = A_i * t^(k_i - 1) * exp(-t / theta_i)

Where A_i is chosen so that sum(C_i) = component_carb_grams (amount per step).

The number of components is `MIXED_MEAL_MIN_COMPONENTS + Poisson(MIXED_MEAL_EXTRA_COMPONENTS_LAMBDA)` capped at `MIXED_MEAL_MAX_COMPONENTS`. Carb fractions per component come from `Dirichlet(MIXED_MEAL_DIRICHLET_ALPHA)`. Each component is sampled as fast / medium / slow with weights driven by `slow_carb_preference`. Within each category, k and theta are uniformly sampled from category-specific ranges (e.g. `MIXED_MEAL_FAST_K_RANGE`).

Per-component noise is applied on top:

    k_actual = k * (1 + N(0, CARB_CURVE_K_NOISE))
    theta_actual = theta * (1 + N(0, CARB_CURVE_THETA_NOISE))

A protein/fat tail (`PROTEIN_FAT_EQUIV_GRAMS` worth of very slow carbs) is always added to every meal regardless of composition.

Hypo correction carbs use a separate fast pair (`HYPO_CARB_K`, `HYPO_CARB_THETA`) that peaks faster than meal carbs (glucose tablets / juice).


## Insulin Action Curves

Bolus (rapid-acting): gamma curve. Both duration and theta scale with dose, centered on a 5U reference:

    sqrt_excess = sqrt(dose) - sqrt(5)
    duration_h = clip(BOLUS_DIA_BASE_HOURS + BOLUS_DIA_DOSE_SCALE * sqrt_excess,
                      BOLUS_DIA_MIN_HOURS, BOLUS_DIA_MAX_HOURS)
    theta = BOLUS_GAMMA_THETA * (1 + BOLUS_THETA_DOSE_SLOPE * sqrt_excess)
    k = BOLUS_GAMMA_K

Larger doses act longer and peak slightly later, matching observed subcutaneous insulin PK. Helper: `bolus_pk_for_dose(dose) -> (k, theta, duration_minutes)`. The legacy `BOLUS_DURATION_HOURS` constant is kept for tests but not used by new code.

Basal (long-acting): trapezoidal curve from `basal_curve()` with `BASAL_RAMP_UP_HOURS` ramp-in, `BASAL_RAMP_DOWN_HOURS` ramp-out, plateau in between, normalized so the area equals the dose.

### Injection site quality (lipohypertrophy)

Each insulin dose (basal, meal bolus, hyper correction, trend correction) is multiplied by a per-dose site_quality factor:

    site_quality ~ N(1.0, SITE_QUALITY_SIGMA_BASE * (1.5 - s4))
    site_quality = clip(site_quality, SITE_QUALITY_MIN, SITE_QUALITY_MAX)
    delivered_dose = intended_dose * site_quality

Patients with low `lifestyle_consistency` (s4) rotate sites poorly and develop lipohypertrophy → higher dose-to-dose variance. The PK shape (k, theta, duration) is determined by the *intended* dose; only the absorbed amount varies.


## Insulin Sensitivity

Multi-peak diurnal pattern (`phase_shift` and `daily_drift` smooth-step-blend across midnight from yesterday to today over `IS_DRIFT_TRANSITION_HOURS`):

    morning = IS_MORNING_AMPLITUDE * exp(-0.5 * ((hour - IS_MORNING_PEAK_HOUR - phase_shift) / 2.0)^2)
    evening = IS_EVENING_AMPLITUDE * exp(-0.5 * ((hour - IS_EVENING_PEAK_HOUR) / 2.5)^2)
    night   = -IS_NIGHT_DIP_AMPLITUDE * exp(-0.5 * ((night_hour - IS_NIGHT_DIP_HOUR) / 2.0)^2)
    diurnal = 1.0 + morning + evening + night

Combined, with all modifiers:

    IS(t) = IS_base * diurnal * (1 + daily_drift) * illness_factor
            * exercise_envelope * stress_envelope
            * glucotox_factor * postprandial_bonus
            * (1 + fast_noise)

Where:
- `daily_drift ~ N(0, IS_DAILY_DRIFT_SIGMA)`, sampled once per day, blended across midnight
- `phase_shift ~ N(0, IS_DAWN_PHASE_DAILY_SIGMA)`, sampled once per day, blended across midnight
- `fast_noise ~ N(0, IS_FAST_NOISE_SIGMA)`, sampled every step
- `illness_factor` ramps toward `illness_is_target` at rate `ILLNESS_IS_RAMP_RATE` per day; always applied (rests at 1.0 when healthy)
- `exercise_envelope`, `stress_envelope` use a trapezoidal `envelope_intensity()` that ramps in/out of the effect window, blending the raw factor against 1.0

### Glucotoxicity

A slow EMA of true BG (6h half-life) drives transient insulin resistance when chronically elevated:

    glucotox_bg_ema += α * (BG - glucotox_bg_ema)   where α = 1 - 0.5^(dt / half_life)
    if glucotox_bg_ema > GLUCOTOX_BG_THRESHOLD:
        intensity = min(1, (ema - threshold) / (max_bg - threshold))
        glucotox_factor = 1 + GLUCOTOX_MAX_IS_INCREASE * intensity
    else:
        glucotox_factor = 1.0

Closes a positive feedback loop: high BG → more IR → harder to bring down.

### Postprandial IS bonus

Incretin / GLP-1 effect: while carbs are absorbing, peripheral tissues are transiently more insulin-sensitive. Saturating in active carb load:

    bonus = POSTPRANDIAL_IS_BONUS_FACTOR * active_carb / (POSTPRANDIAL_IS_BONUS_HALF + active_carb)
    postprandial_bonus = 1 - bonus


## BG Delta Computation

At each step:

    glucose_in  = total_carb + HGO - exercise
    glucose_out = total_insulin * ICR / IS(t)
    delta_BG    = BG_SCALE_FACTOR * (glucose_in - glucose_out)

`IS(t)` now divides the insulin side: insulin-resistant patients (IS > 1) clear less glucose per unit insulin; sensitive patients (IS < 1) clear more. HGO suppression by insulin is handled separately (see Hepatic Glucose Output) — IS only modulates peripheral insulin action.

Physiological guardrails:

    if BG > RENAL_THRESHOLD:
        delta_BG -= (BG - RENAL_THRESHOLD) * RENAL_CLEARANCE_RATE

    if BG < COUNTER_REGULATORY_THRESHOLD:
        delta_BG += COUNTER_REGULATORY_RATE * (COUNTER_REGULATORY_THRESHOLD - BG) / COUNTER_REGULATORY_THRESHOLD

    if BG < SEVERE_HYPO_THRESHOLD:
        severity = (SEVERE_HYPO_THRESHOLD - BG) / SEVERE_HYPO_THRESHOLD
        delta_BG += SEVERE_HYPO_GLUCAGON_RATE * severity

Final:

    BG(t+1) = clamp(BG(t) + delta_BG, BG_CLAMP_MIN, BG_CLAMP_MAX)

`BG_CLAMP_MIN` is intentionally low (20 mg/dL) so the dynamics, not the clamp, drive the lower tail. The combined counter-regulatory + glucagon-dump terms together with the soft-bound headroom cap are usually strong enough to lift BG before the clamp engages.


## CGM Observation Model

    BG_observed = BG_true + N(0, sigma_cgm)
    sigma_cgm = CGM_NOISE_FRACTION * BG_true

This gives proportional noise: higher BG = more absolute noise, matching real CGM MARD characteristics.

Interstitial lag is currently NOT modeled. The constant `CGM_LAG_MINUTES = 10` is reserved for a future implementation that would sample true BG from `CGM_LAG_MINUTES` ago instead of the current step. Real CGMs lag by 5-15 min; ML models trained on this simulator's CGM channel will not learn that lag.


## Hepatic Glucose Output

Insulin-suppressed via a Hill function on EMA-smoothed insulin (proxies plasma insulin lag behind subcutaneous absorption, ~12 min half-life at `HGO_INSULIN_SMOOTHING_ALPHA = 0.25`):

    smoothed_ins  = α * insulin_per_step + (1-α) * smoothed_ins_prev
    suppression   = 1 / (1 + smoothed_ins / HGO_INSULIN_HALF_MAX)
    HGO_rate      = HGO_SUPPRESSED_FLOOR + (HGO_UNSUPPRESSED - HGO_SUPPRESSED_FLOOR) * suppression
    HGO_baseline  = HGO_rate * (1 + N(0, HGO_NOISE_SIGMA)) * (DT_MINUTES / 60) * glycogen_gate * alcohol_factor
    meal_rebound  = sum over active meal_hgo_effects of (magnitude * envelope_intensity) * (DT_MINUTES / 60)
    HGO(t)        = HGO_baseline + meal_rebound

`HGO_INSULIN_HALF_MAX` is tuned so a typical basal level (~0.07 U/step) yields ~9 g/hr (the legacy balanced rate, preserved so basal sizing — `ideal_basal = HGO_BASE_GRAMS_PER_HOUR * 24 / ICR` — still produces near-zero net delta). At zero insulin, HGO climbs toward `HGO_UNSUPPRESSED_GRAMS_PER_HOUR` (DKA-like). Alcohol additionally suppresses HGO via `alcohol_factor` (trapezoidal envelope around 1.0). The `glycogen_gate` ramps HGO down when the reservoir is depleted (see Glycogen reservoir). The `meal_rebound` term is additive (not multiplicative) — see Delayed-meal HGO rebound below.

Helper: `compute_hgo_rate(insulin_per_step) -> g/hr`.

### Delayed-meal HGO rebound

Each meal whose carb amount exceeds `DELAYED_HGO_MEAL_THRESHOLD_GRAMS` schedules a positive HGO bump 3.5-5.5h later, lasting 4-8h:

    excess     = carb_amount - DELAYED_HGO_MEAL_THRESHOLD_GRAMS
    magnitude  = min(DELAYED_HGO_MAX_BUMP, DELAYED_HGO_PER_GRAM * excess)   (g/hr)
    delay      ~ U(DELAYED_HGO_DELAY_HOURS_MIN, DELAYED_HGO_DELAY_HOURS_MAX)
    duration   ~ U(DELAYED_HGO_DURATION_HOURS_MIN, DELAYED_HGO_DURATION_HOURS_MAX)

The bump is shaped by the standard trapezoidal `envelope_intensity()` with `DELAYED_HGO_RAMP_HOURS` ramps. Models the delayed gluconeogenesis from amino acids and cortisol response that drive nocturnal hyperglycemia after a large dinner.

### Glycogen reservoir

Hepatic glycogen is a finite store that gates the glycogenolysis-sourced fraction of HGO:

    if glycogen < GLYCOGEN_CAPACITY * GLYCOGEN_LOW_THRESHOLD_FRACTION:
        availability = glycogen / (GLYCOGEN_CAPACITY * GLYCOGEN_LOW_THRESHOLD_FRACTION)
        glycogen_gate = (1 - GLYCOGEN_DRAIN_FRACTION) + GLYCOGEN_DRAIN_FRACTION * availability
    else:
        glycogen_gate = 1.0

After applying the gate, glycogen is updated each step:

    glycogen += total_carb * GLYCOGEN_REFILL_FRACTION   (refill from absorbed carbs)
    glycogen -= HGO(t) * GLYCOGEN_DRAIN_FRACTION        (drain from glycogenolysis)
    glycogen  = clip(glycogen, 0, GLYCOGEN_CAPACITY)

The refill is a "background" channel — it is not subtracted from BG-bound carbs, since ICR is empirically tuned to net BG response. The coupling back to BG dynamics is via the `glycogen_gate` reducing future HGO when the reservoir is depleted.


## Correction Behavior

Hypo correction (BG_observed < BG_LOW_THRESHOLD):

    correction_grams = HYPO_CORRECTION_BASE_GRAMS + panic_factor * severity / 20
    severity = max(0, BG_LOW_THRESHOLD - BG_observed)

If BG_observed < RAGE_EAT_BG_THRESHOLD, rage eating may occur with probability proportional to (1.2 - dosing_competence).

Hyper correction (BG_observed > BG_HIGH_THRESHOLD):

    correction_dose = (BG_observed - BG_TARGET) / correction_factor * (1 + noise)
    patience = patience_time / urgency
    urgency = max(1, (BG_observed - 250) / 50) if BG > 250, else 1

If BG_observed > RAGE_BOLUS_BG_THRESHOLD, rage bolusing may occur.


## Basal Adjustment

Daily adjustment based on previous day's mean BG:

    if mean_BG > 150:
        overshoot = min((mean_BG - 150) / 100, 1)
        adjustment = 1 + overshoot * BASAL_CORRECTION_MAX_ADJUSTMENT * competence

    if mean_BG < 90:
        undershoot = min((90 - mean_BG) / 50, 1)
        adjustment = 1 - undershoot * BASAL_CORRECTION_MAX_ADJUSTMENT * competence


## Unit Conventions

All curve values are in "amount per step" units:
- Carb curves: grams per step (sum of curve = total grams)
- Insulin curves: units per step (sum of curve = total units)
- HGO: grams per step (rate g/hr converted via DT_MINUTES / 60)
- Exercise: grams-equivalent per step

Both `gamma_curve` and `basal_curve` normalize so that `sum(values) = total_amount`. There is no `flat_curve` — the trapezoidal `basal_curve` replaced it. Never pass a rate where total_amount is expected.
