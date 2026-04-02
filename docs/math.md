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

Each meal produces a gamma-distributed absorption curve:

    C(t) = A * t^(k-1) * exp(-t/theta)

Where A is chosen so that sum(C) = total_carb_grams (amount per step).

Parameters by carb type:
- Fast carbs: k = FAST_CARB_K, theta = FAST_CARB_THETA
- Slow carbs: k = SLOW_CARB_K, theta = SLOW_CARB_THETA
- Protein/fat: k = PROTEIN_FAT_GAMMA_K, theta = PROTEIN_FAT_GAMMA_THETA

Noise is applied to k and theta per meal:

    k_actual = k * (1 + N(0, CARB_CURVE_K_NOISE))
    theta_actual = theta * (1 + N(0, CARB_CURVE_THETA_NOISE))


## Insulin Action Curves

Bolus (rapid-acting): gamma curve with k = BOLUS_GAMMA_K, theta = BOLUS_GAMMA_THETA.
Duration: BOLUS_DURATION_HOURS * 60 minutes.

Basal (long-acting): flat curve over BASAL_DURATION_HOURS.
Rate per step = dose / (duration_hours) * (DT_MINUTES / 60).


## Insulin Sensitivity

Multi-peak diurnal pattern:

    morning = IS_MORNING_AMPLITUDE * exp(-0.5 * ((hour - IS_MORNING_PEAK_HOUR - phase_shift) / 2.0)^2)
    evening = IS_EVENING_AMPLITUDE * exp(-0.5 * ((hour - IS_EVENING_PEAK_HOUR) / 2.5)^2)
    night   = -IS_NIGHT_DIP_AMPLITUDE * exp(-0.5 * ((night_hour - IS_NIGHT_DIP_HOUR) / 2.0)^2)
    diurnal = 1.0 + morning + evening + night

Combined:

    IS(t) = IS_base * diurnal * (1 + daily_drift) * illness_factor * (1 + fast_noise)

Where:
- daily_drift ~ N(0, IS_DAILY_DRIFT_SIGMA), sampled once per day
- phase_shift ~ N(0, IS_DAWN_PHASE_DAILY_SIGMA), sampled once per day
- fast_noise ~ N(0, IS_FAST_NOISE_SIGMA), sampled every step
- illness_factor ramps toward illness_is_target at rate ILLNESS_IS_RAMP_RATE per day


## BG Delta Computation

At each step:

    effective_carb_load = (total_carb + HGO - exercise) * IS(t)
    insulin_carb_equiv = total_insulin * ICR
    delta_BG = BG_SCALE_FACTOR * (effective_carb_load - insulin_carb_equiv)

Physiological guardrails:

    if BG > RENAL_THRESHOLD:
        delta_BG -= (BG - RENAL_THRESHOLD) * RENAL_CLEARANCE_RATE

    if BG < COUNTER_REGULATORY_THRESHOLD:
        delta_BG += COUNTER_REGULATORY_RATE * (threshold - BG) / threshold

Final:

    BG(t+1) = clamp(BG(t) + delta_BG, BG_CLAMP_MIN, BG_CLAMP_MAX)


## CGM Observation Model

    BG_observed = BG_true + N(0, sigma_cgm)
    sigma_cgm = CGM_NOISE_FRACTION * BG_true

This gives proportional noise: higher BG = more absolute noise, matching real CGM MARD characteristics.


## Hepatic Glucose Output

Constant rate with noise:

    HGO(t) = HGO_BASE_GRAMS_PER_HOUR * (1 + N(0, HGO_NOISE_SIGMA)) * (DT_MINUTES / 60)


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
- HGO: grams per step
- Exercise: grams-equivalent per step

The gamma_curve function normalizes so that sum(values) = total_amount.
The flat_curve function computes rate_per_hour * (DT_MINUTES / 60).
Both produce values in the same units (amount per step).
