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

A protein/fat tail is always added to every meal regardless of composition, scaled to meal size as `clip(PROTEIN_FAT_FRACTION_OF_CARBS * carb_amount, PROTEIN_FAT_MIN_GRAMS, PROTEIN_FAT_MAX_GRAMS)`. With the current floor (`PROTEIN_FAT_MIN_GRAMS = 6 g`), snacks get ~6 g, typical meals ~10–15 g, large dinners ~18 g.

Hypo correction carbs use a separate fast pair (`HYPO_CARB_K`, `HYPO_CARB_THETA`) that peaks faster than meal carbs (glucose tablets / juice).


## Insulin Action Curves

Bolus (rapid-acting): gamma curve. Both duration and theta scale with dose, centered on a 5U reference:

    sqrt_excess = sqrt(dose) - sqrt(5)
    duration_h = clip(BOLUS_DIA_BASE_HOURS + BOLUS_DIA_DOSE_SCALE * sqrt_excess,
                      BOLUS_DIA_MIN_HOURS, BOLUS_DIA_MAX_HOURS)
    theta = BOLUS_GAMMA_THETA * (1 + BOLUS_THETA_DOSE_SLOPE * sqrt_excess)
    k = BOLUS_GAMMA_K

Larger doses act longer and peak slightly later, matching observed subcutaneous insulin PK. Helper: `bolus_pk_for_dose(dose) -> (k, theta, duration_minutes)`. The legacy `BOLUS_DURATION_HOURS` constant is kept for tests but not used by new code.

Basal (long-acting): Bateman one-compartment PK curve from `basal_curve()` —

    f(t) = exp(-BASAL_KE_PER_HOUR · t) − exp(-BASAL_KA_PER_HOUR · t)

modelling subcutaneous depot absorption (rate `ka`) followed by first-order elimination (rate `ke`). With the default `ka = 0.30/h` and `ke = 0.07/h` the curve rises smoothly from zero, peaks at `tmax = ln(ka/ke)/(ka − ke) ≈ 6.3 h` post-injection, and then decays with a ~9.9 h elimination half-life — a broad-peaked long-acting profile sitting between the glargine and degludec time-action curves. There is no flat plateau and no slope discontinuity. A smootherstep window over the last `BASAL_TAIL_CLIP_HOURS` tapers the late residual to zero so consecutive daily doses join without a tail-step. Normalized so the area equals the dose.

### Injection site quality (lipohypertrophy)

Each insulin dose (basal, meal bolus, hyper correction, trend correction) is multiplied by a per-dose site_quality factor:

    site_quality ~ N(1.0, SITE_QUALITY_SIGMA_BASE * (1.5 - s4) ** 1.8)
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
            * glucotox_factor * postprandial_ir
            * (1 + fast_noise)

Where:
- `daily_drift ~ N(0, IS_DAILY_DRIFT_SIGMA * (1.5 - s4))`, sampled once per day, blended across midnight — the drift std scales with `(1.5 - lifestyle_consistency)`, so consistent-lifestyle patients swing less day-to-day
- `phase_shift ~ N(0, IS_DAWN_PHASE_DAILY_SIGMA)`, sampled once per day, blended across midnight
- `fast_noise ~ N(0, IS_FAST_NOISE_SIGMA)`, sampled every step
- `illness_factor` ramps toward `illness_is_target` at rate `ILLNESS_IS_RAMP_RATE` per day; always applied (rests at 1.0 when healthy)
- `exercise_envelope`, `stress_envelope` use a trapezoidal `envelope_intensity()` that ramps in/out of the effect window, blending the raw factor against 1.0

### Glucotoxicity

A slow EMA of true BG (3h half-life) drives transient insulin resistance when chronically elevated:

    glucotox_bg_ema += α * (BG - glucotox_bg_ema)   where α = 1 - 0.5^(dt / half_life)
    if glucotox_bg_ema > GLUCOTOX_BG_THRESHOLD:
        intensity = min(1, (ema - threshold) / (max_bg - threshold))
        glucotox_factor = 1 + GLUCOTOX_MAX_IS_INCREASE * intensity
    else:
        glucotox_factor = 1.0

Closes a positive feedback loop: high BG → more IR → harder to bring down.

### Postprandial insulin resistance

In non-diabetics the incretin / GLP-1 axis augments insulin secretion and sensitivity around a meal. In T1DM that axis is blunted and there is no endogenous insulin response, so the meal-time sensitivity boost is absent; if anything the absorbing-carb state is mildly insulin-*resistant*. While carbs are absorbing, IR is raised (insulin clears glucose slightly less effectively). Saturating in active carb load:

    penalty = POSTPRANDIAL_IR_PENALTY_FACTOR * active_carb / (POSTPRANDIAL_IR_PENALTY_HALF + active_carb)
    postprandial_ir = 1 + penalty


## BG Delta Computation

At each step:

    glucose_in  = total_carb + HGO - exercise
    glucose_out = total_insulin * ICR / IS(t)
    delta_BG    = BG_SCALE_FACTOR * (glucose_in - glucose_out)
    delta_BG   += Sg * (E(t) - BG)     # glucose-effectiveness restoring pull (see below)

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

`BG_CLAMP_MIN` is set to 40 mg/dL, matching the real CGM device floor (the Ohio/AZT1D minimum). The combined counter-regulatory + glucagon-dump terms together with the soft-bound headroom cap are usually strong enough to lift BG before the clamp engages.

### Glucose effectiveness (Bergman Sg) equilibrium

The restoring term `delta_BG += Sg * (E(t) - BG)` is an always-on, insulin-independent pull toward a stochastic equilibrium `E(t)`. It supplies the within-band mean reversion the renal / counter-regulatory guardrails do not; inside 70–180 net flux is otherwise integrated with an over-long autocorrelation. `Sg = glucose_effectiveness` is the per-patient Bergman minimal-model glucose effectiveness (a per-step reversion fraction), sampled lognormally around `GE_RATE` and clipped to `[GE_RATE_MIN, GE_RATE_MAX]` (~2–3× inter-individual spread; the floor prevents a pure integrator).

`E(t)` is an Ornstein–Uhlenbeck process. With `rho = exp(-DT_MINUTES / (GE_EQ_TAU_HOURS * 60))` it mean-reverts each step toward a target `mu`, is perturbed by Gaussian noise, then floored:

    mu = ge_anchor + ge_dawn_amplitude * ge_diurnal_profile(hour)
    E  = mu + rho * (E_prev - mu) + sqrt(1 - rho^2) * GE_EQ_SIGMA * ge_sigma_mult * N(0, 1)
    E  = max(E, GE_EQ_FLOOR)

The `sqrt(1 - rho^2)` factor makes the stationary std equal `GE_EQ_SIGMA * ge_sigma_mult`. The strong, fast Sg pull gives BG a short correlation time (8h ACF ≈ 0) while `E`'s wandering supplies distributional spread that decorrelates within hours, decoupling spread from the autocorrelation. `GE_EQ_FLOOR = 64` sits above `SEVERE_HYPO_THRESHOLD = 55`, so the pull is always upward in a severe low (it aids, never opposes, the rescue). `ge_diurnal_profile(hour)` is a mean-zero wrapped-Gaussian dawn-phenomenon rhythm peaking at `GE_DAWN_PEAK_HOUR = 8` with width `GE_DAWN_WIDTH_HOURS = 5.5`, mean-subtracted over the 24h day so it adds rhythm without shifting the pooled mean; its per-patient amplitude `ge_dawn_amplitude` scales with the same dawn trait as the HGO surge.

Per-patient heterogeneity (sampled once in `generate_patient`):

    ir            = clip(exp(N(0, IR_LOGNORMAL_SIGMA)), IR_FACTOR_MIN, IR_FACTOR_MAX)
    ge_anchor     = clip(N(GE_EQ_ANCHOR_MEAN + GE_ANCHOR_IR_COUPLING * (ir - 1), GE_EQ_ANCHOR_SIGMA), 110, 210)
    ge_sigma_mult = clip(exp(N(0, GE_SIGMA_REL_SIGMA)), GE_SIGMA_MULT_CLIP)

with `IR_LOGNORMAL_SIGMA = 0.26`, `IR_FACTOR_MIN / IR_FACTOR_MAX = 0.4 / 2.0`, `GE_EQ_ANCHOR_MEAN = 138`, `GE_ANCHOR_IR_COUPLING = 30` (mg/dL per unit `ir - 1`), `GE_EQ_ANCHOR_SIGMA = 15`, `GE_SIGMA_REL_SIGMA = 0.16`, and `GE_SIGMA_MULT_CLIP = (0.68, 1.38)`. The anchor's between-patient spread carries the per-patient mean-glucose heterogeneity; the insulin-resistance coupling raises the anchor for resistant patients (higher `ir` → higher mean glucose) on the high side that `GE_EQ_FLOOR` does not compress. The per-patient `ge_sigma_mult` makes patients differ in *within*-patient variability rather than sharing one global `GE_EQ_SIGMA`. The same `ir` also seeds `is_base`, `icr`, and `correction_factor`.


## CGM Observation Model

The sensor reports a delayed-and-smoothed interstitial value with time-correlated multiplicative noise — never the instantaneous true BG. First a first-order interstitial lag (Rebrin/Steil), then AR(1) sensor noise applied multiplicatively:

    alpha_lag   = 1 - exp(-DT_MINUTES / CGM_LAG_MINUTES)
    IG         += alpha_lag * (BG_true - IG)
    ar_cgm      = NOISE_AR1_RHO_SENSOR * ar_cgm + NOISE_AR1_INNOV_SENSOR * N(0, CGM_NOISE_FRACTION)
    BG_observed = IG * (1 + ar_cgm)

Because the noise multiplies the reading, its std scales with BG: higher BG = more absolute noise, matching real CGM MARD characteristics. The AR(1) coefficient `NOISE_AR1_RHO_SENSOR = 0.92` (~42 min half-life, with `NOISE_AR1_INNOV_SENSOR = sqrt(1 - 0.92^2)`) produces smoothly-drifting offsets over 30-60 min windows rather than white-noise spikes.

Interstitial lag is modeled as first-order diffusion with timescale `CGM_LAG_MINUTES = 15` (applied before the sensor noise), so every consumer of `BG_observed` — corrections, hypo detection, the exported CGM channel — sees the delayed-and-smoothed value, not the current step.


## Hepatic Glucose Output

Insulin-suppressed via a Hill function on EMA-smoothed insulin (proxies plasma insulin lag behind subcutaneous absorption, ~12 min half-life at `HGO_INSULIN_SMOOTHING_ALPHA = 0.25`):

    smoothed_ins  = α * insulin_per_step + (1-α) * smoothed_ins_prev
    suppression   = 1 / (1 + smoothed_ins / HGO_INSULIN_HALF_MAX)
    HGO_rate      = HGO_SUPPRESSED_FLOOR + (HGO_UNSUPPRESSED - HGO_SUPPRESSED_FLOOR) * suppression
    HGO_baseline  = max(0, HGO_rate * (1 + N(0, HGO_NOISE_SIGMA)) * (body_weight_kg / BODY_WEIGHT_MEAN_KG) * (DT_MINUTES / 60)
                            + (dawn_g_per_hr - night_dip_g_per_hr) * (DT_MINUTES / 60)) * glycogen_gate * alcohol_factor
    meal_rebound  = sum over active meal_hgo_effects of (magnitude * envelope_intensity) * (DT_MINUTES / 60)
    HGO(t)        = HGO_baseline + meal_rebound

`HGO_INSULIN_HALF_MAX` is tuned so a typical basal level (~0.086 U/step) yields 8.25 g/hr (`HGO_BASE_GRAMS_PER_HOUR`, the balanced reference rate, preserved so basal sizing — `ideal_basal = HGO_BASE_GRAMS_PER_HOUR * 24 * (body_weight_kg / BODY_WEIGHT_MEAN_KG) * is_base / ICR` — still produces near-zero net delta. The weight factor mirrors the per-step HGO scaling and the `is_base` factor keeps the invariant across baseline insulin needs). At zero insulin, HGO climbs toward `HGO_UNSUPPRESSED_GRAMS_PER_HOUR` (DKA-like). The additive `(dawn_g_per_hr - night_dip_g_per_hr)` term is a cortisol-driven dawn surge (Gaussian peaking at `DAWN_HGO_PEAK_HOUR`) minus a deep-sleep trough (Gaussian at `NIGHT_HGO_DIP_HOUR`), added in g/hr rather than as a multiplier so the Hill insulin suppression does not cancel it — this is what produces the dawn phenomenon. Alcohol additionally suppresses HGO via `alcohol_factor` (trapezoidal envelope around 1.0). The `glycogen_gate` ramps HGO down when the reservoir is depleted (see Glycogen reservoir). The `meal_rebound` term is additive (not multiplicative) — see Delayed-meal HGO rebound below.

Helper: `compute_hgo_rate(insulin_per_step) -> g/hr`.

### Delayed-meal HGO rebound

Each meal whose carb amount exceeds `DELAYED_HGO_MEAL_THRESHOLD_GRAMS` schedules a positive HGO bump 3.5-5.5h later, lasting 2.7-7.0h:

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

    glycogen -= HGO(t) * GLYCOGEN_DRAIN_FRACTION        (drain from glycogenolysis)
    glycogen += total_carb * GLYCOGEN_REFILL_FRACTION   (refill from absorbed carbs)
    glycogen  = clip(glycogen, 0, GLYCOGEN_CAPACITY)

The refill is a "background" channel — it is not subtracted from BG-bound carbs, since ICR is empirically tuned to net BG response. The coupling back to BG dynamics is via the `glycogen_gate` reducing future HGO when the reservoir is depleted.


## Correction Behavior

Hypo correction (BG_observed < eff_low_thresh):

    skill_avg         = (attentiveness + dosing_competence) / 2
    eff_low_thresh    = BG_LOW_THRESHOLD + 18 * skill_avg
    skill_multiplier  = 1 + 1.5 * skill_avg
    correction_grams  = HYPO_CORRECTION_BASE_GRAMS * skill_multiplier
                        + panic_factor * severity / 20
    severity          = max(0, eff_low_thresh - BG_observed)

The trigger and severity are measured against the skill-scaled `eff_low_thresh`, not the raw `BG_LOW_THRESHOLD`: attentive/competent patients act on the drop before crossing 70 (for `skill_avg = 0.7` the trigger lands near 73).

The skill multiplier is critical — without it, high-skill patients linger at TBR ~30% because the bare base grams (~6 g) cannot overcome a strong basal pipeline. With it, skilled patients reliably exit hypo while unskilled patients still under-correct.

For severe hypo (BG_observed < `SEVERE_HYPO_THRESHOLD`, default 55):

    deficit = SEVERE_HYPO_THRESHOLD - BG_observed
    correction_grams = max(correction_grams, 14 + 0.35 * deficit)

This is the non-probabilistic rage-eat that keeps severe episodes under 1h. Severe hypo also bypasses the CGM check interval, but a `SEVERE_HYPO_REFRACTORY_MIN` (10 min) gate still applies between back-to-back doses so stacked carbs don't sawtooth into post-correction hypers. After any hypo correction, basal is scaled down by `POST_HYPO_BASAL_SUSPEND_FACTOR` for `POST_HYPO_BASAL_SUSPEND_DURATION_HOURS` (pump-suspend / temp-basal analogue), and skill-gated patients (`skill_avg > HYPO_FOLLOWUP_SKILL_THRESHOLD`) eat a slow-carb follow-up snack sized as `HYPO_FOLLOWUP_FRACTION × correction_grams`.

If BG_observed < RAGE_EAT_BG_THRESHOLD, rage eating may occur with probability proportional to (1.2 - dosing_competence).

Hyper correction (BG_observed > eff_high_thresh):

    eff_high_thresh   = BG_HIGH_THRESHOLD - 25 * skill_avg
    iob_consideration = IOB * correction_factor * (0.7 + 0.3 * dosing_competence)
    adjusted_excess   = max(0, (BG_observed - BG_TARGET) - iob_consideration)
    correction_dose   = max(0.5, adjusted_excess / correction_factor * (1 + noise))
    urgency           = min(3, 1 + max(0, (BG_observed - BG_HIGH_THRESHOLD) / 50))
    patience          = patience_time / urgency

The dose is IOB-aware: an insulin-on-board term (a baseline 70% of the expected IOB drop plus a skill-scaled bonus up to 30%) is subtracted before sizing, so patients don't stack corrections. Urgency ramps from the correction threshold `BG_HIGH_THRESHOLD` upward and saturates at 3 (reached at BG = 275), shortening the patience window for sustained highs.

If BG_observed > RAGE_BOLUS_BG_THRESHOLD, rage bolusing may occur.


## Basal Adjustment

Daily adjustment based on a 3-day rolling mean BG (`recent_mean`):

    if recent_mean > 150:
        overshoot     = min((recent_mean - 150) / 80, 1)
        skill_factor  = 0.4 + 0.6 * competence
        trigger_mean  = max(recent_mean, one_day_mean)
        extreme_boost = 1 + 0.5 * min(1, (trigger_mean - 200) / 50)   if trigger_mean > 200, else 1
        adjustment    = 1 + overshoot * (BASAL_CORRECTION_MAX_ADJUSTMENT * skill_factor) * extreme_boost

    elif recent_mean < 115:
        undershoot = min((115 - recent_mean) / 50, 1)
        adjustment = 1 - undershoot * BASAL_CORRECTION_MAX_ADJUSTMENT * competence

The asymmetric thresholds (150 / 115) intentionally bias toward correcting persistent hyperglycemia faster than persistent mild hypoglycemia. On the high path skill scales only partially (a 40% baseline plus up to 60% more), and `extreme_boost` accelerates recovery when the 3-day or single-day mean runs above 200; the low path keeps full skill scaling.


## Behavioral & Stochastic Features

Mechanisms that perturb the deterministic core above. Each closes the gap between an idealized model and the messy reality of a free-living patient.

### Soft-bound BG headroom cap

In addition to the hard clamp at `BG_CLAMP_MIN` / `BG_CLAMP_MAX`, each step's `delta_BG` is capped so it can only close a fraction of the headroom remaining to the soft bound. Lets the dynamics asymptote toward the bounds instead of slamming into the clamp:

    projected = BG(t) + delta_BG
    if projected < BG_SOFT_FLOOR:
        headroom = max(0, BG(t) - BG_CLAMP_MIN)
        delta_BG  = max(delta_BG, -SOFT_APPROACH_FRACTION * headroom)
    if projected > BG_SOFT_CEILING:
        headroom = max(0, BG_CLAMP_MAX - BG(t))
        delta_BG  = min(delta_BG,  SOFT_APPROACH_FRACTION * headroom)

The hard clamp is still applied as a backstop after this cap.

### Per-step absorption noise

Multiplicative noise applied to the per-step carb and insulin contributions read from the accumulation arrays:

    total_carb    *= max(0, 1 + N(0, CARB_ABSORPTION_NOISE_SIGMA))      if total_carb > 0
    total_insulin *= max(0, 1 + N(0, INSULIN_ABSORPTION_NOISE_SIGMA))   if total_insulin > 0

Models moment-to-moment variation in absorption that the smooth gamma curves cannot capture. Only active when the underlying curve is non-zero.

### Exercise post-effect IS envelope

After an exercise event ends, IS is reduced (more sensitive) for `EXERCISE_IS_DURATION_HOURS` (6h), shaped by the trapezoidal `envelope_intensity()` with `EXERCISE_IS_RAMP_HOURS` ramps:

    reduction  = min(0.30, EXERCISE_IS_REDUCTION * (exercise_duration / EXERCISE_DURATION_MEAN_MIN))
    factor(t)  = 1 - reduction * envelope_intensity(t; start, start+6h, ramp=1h)
    exercise_envelope(t) = factor(t)

Larger / longer sessions produce a stronger reduction, capped at 30%.

### Stress IS envelope

Stress events transiently raise IS (more resistant) for `STRESS_DURATION_HOURS_MIN..MAX` hours:

    is_factor      ~ U(STRESS_IS_FACTOR_MIN, STRESS_IS_FACTOR_MAX)
    duration       ~ U(STRESS_DURATION_HOURS_MIN, STRESS_DURATION_HOURS_MAX)
    stress_envelope(t) = 1 + (is_factor - 1) * envelope_intensity(t; start, end, STRESS_IS_RAMP_HOURS)

Per-day probability scales with `STRESS_PROBABILITY_BASE - STRESS_LIFESTYLE_WEIGHT * s4`.

### Alcohol HGO suppression envelope

Drinking suppresses HGO multiplicatively, with an onset delay, plateau, and ramp-down:

    hgo_reduction ~ U(ALCOHOL_HGO_REDUCTION_MIN, ALCOHOL_HGO_REDUCTION_MAX)
    duration      ~ U(ALCOHOL_DURATION_HOURS_MIN, ALCOHOL_DURATION_HOURS_MAX)
    onset_delay   ~ U(ALCOHOL_ONSET_DELAY_HOURS_MIN, ALCOHOL_ONSET_DELAY_HOURS_MAX)
    alcohol_factor(t) = 1 - hgo_reduction * envelope_intensity(t; start+onset, start+onset+duration,
                                                                ALCOHOL_HGO_RAMP_HOURS)

This multiplies the Hill-derived HGO baseline (separate from insulin suppression), explaining the nocturnal-hypo pattern after evening drinking.

### Trend-based anticipatory corrections

Attentive patients with sufficient skill act on a recent BG trend before crossing a threshold. From a sliding window of the last `TREND_CORRECTION_WINDOW_STEPS` BG samples:

    trend = (window[-1] - window[0]) / (TREND_CORRECTION_WINDOW_STEPS - 1)   (mg/dL/step)

A preemptive correction bolus is considered when `trend > TREND_HIGH_RATE_THRESHOLD` and BG is approaching the upper band; a preemptive snack is considered when `trend < TREND_LOW_RATE_THRESHOLD` and BG is approaching the lower band. The projected rise/fall over the next `2 * TREND_CORRECTION_WINDOW_STEPS` steps sizes the dose / carbs.

### Anomalous absorption events

With per-day probability `ANOMALOUS_EVENT_PROBABILITY`, one absorption curve on that day has its shape modified:

    k     *= U(ANOMALOUS_K_MULT_MIN,     ANOMALOUS_K_MULT_MAX)
    theta *= U(ANOMALOUS_THETA_MULT_MIN, ANOMALOUS_THETA_MULT_MAX)

Models unusual gastric emptying, food composition outliers, or other absorption surprises.

### Rare event days

With per-day probability `RARE_EVENT_PROBABILITY`, three of the four skills — dietary discipline (s1), dosing competence (s3), and lifestyle consistency (s4) — are degraded for that day (attentiveness s2 is left unchanged):

    skill_penalty ~ RARE_EVENT_SKILL_REDUCTION + U(0, 0.3)
    s_i(today)    = max(s_i - skill_penalty, 0.05)   for i in {1, 3, 4}

Even attentive, well-controlled patients have chaotic days (illness onset, travel, emotional events). This is the simulator's way of injecting irreducible behavioral noise.


## Unit Conventions

All curve values are in "amount per step" units:
- Carb curves: grams per step (sum of curve = total grams)
- Insulin curves: units per step (sum of curve = total units)
- HGO: grams per step (rate g/hr converted via DT_MINUTES / 60)
- Exercise: grams-equivalent per step

Both `gamma_curve` and `basal_curve` normalize so that `sum(values) = total_amount`. There is no `flat_curve` — `basal_curve` (Bateman PK, smooth onset/peak/decline) replaced it. Never pass a rate where total_amount is expected.
