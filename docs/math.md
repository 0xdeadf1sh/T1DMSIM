# Mathematical Formulation

Reference for the T1DM simulator's model: curve generation, the BG delta pipeline, and the behavioral layers that perturb it.


## Per-Step Pipeline

```mermaid
flowchart TD
    CARB["carb curves<br/>meals · hypo corrections · follow-up snacks"] --> GIN["glucose_in = carb + HGO"]
    CARB -.->|"guessed grams and GI"| LOG["logged carb curve<br/>exported as total_carb; drives nothing"]
    HGO["HGO<br/>Hill insulin suppression × glycogen gate × alcohol<br/>+ dawn/night term + delayed-meal rebound"] --> GIN
    INS["insulin curves<br/>basal · bolus · corrections<br/>× injection-site quality"] --> GOUT["glucose_out = insulin × ICR / IS"]
    INS -.->|"intended dose"| INSLOG["logged insulin curve<br/>exported as total_insulin; drives nothing"]
    IS["IS(t)<br/>diurnal × daily drift × illness × stress<br/>× glucotoxicity × postprandial IR × fast noise"] --> GOUT
    GIN --> DELTA["delta_BG = BG_SCALE_FACTOR × (glucose_in − glucose_out)<br/>+ Sg × (E(t) − BG), the OU equilibrium pull"]
    GOUT --> DELTA
    DELTA --> GUARD["guardrails<br/>renal clearance · counter-regulatory · glucagon dump"]
    GUARD --> BOUND["soft headroom cap → hard clamp → BG(t+1)"]
    BOUND --> CGM["CGM model<br/>interstitial lag → AR(1) noise → BG_observed"]
    CGM --> BEH["patient behavior<br/>meals · boluses · hypo/hyper corrections · basal adjust"]
    BEH -.-> CARB
    BEH -.-> INS
```


## Patient Skill Profile

Sample from a 4D multivariate normal — `Sigma` has `SKILL_VARIANCE` on the diagonal and `SKILL_CORRELATION * SKILL_VARIANCE` off-diagonal — then squash and clamp:

    s_raw = (s1, s2, s3, s4) ~ N(0, Sigma)
    s_i   = sigmoid(s_raw_i) = 1 / (1 + exp(-s_raw_i))
    s_i   = clip(s_i, SKILL_MIN, SKILL_MAX)


## Carbohydrate Absorption Curves

Every carbohydrate curve is a gamma whose shape comes from a glycaemic index (`gi_gamma_params`, SPEC/invariants.md §5). Entry `n` of a curve is the amount appearing *during* step `n`, so `gamma_curve` integrates the gamma across the step — the mean of `GAMMA_CURVE_SUBSTEPS = 16` sub-step midpoints, starting from `t = 0`:

    C[n] = A * w[n] * mean{ t^(k - 1) * exp(-t / theta) : t in [n*dt, (n+1)*dt) }
           A s.t. sum_n C[n] = carb_grams

Averaging the density across the step is what sizes the leading edge: the first step of a `k = 2, theta = 15` rescue curve carries 37% of the peak and a `k = 3, theta = 25` bolus 2%, so a curve rises through its onset. `w[n]` is 1 except over the last `CURVE_TAIL_CLIP_FRACTION` (1/6) of the steps, where a smootherstep takes it to exactly 0, so no curve ends on a cliff. The same construction applies to every gamma curve — meals, rescue and follow-up carbs, and every bolus.

    g        = clip(GI, 0, 100) / 100
    k        = 4.5 + (2.0 - 4.5) * g
    theta    = 30.0 + (15.0 - 30.0) * g
    duration = clip(4 * k * theta, 120, 360)                    minutes

    meal_gi_mean = MEAL_GI_MEAN_MAX - MEAL_GI_DISCIPLINE_SPAN * s1       per patient
    meal GI      = clip(N(meal_gi_mean, MEAL_GI_SIGMA), 0, 100)          per meal

One curve per meal, summing to the meal's grams, `carb_amount`. Rescue and trend carbs use `RESCUE_CARB_GI` (100); the hypo follow-up snack uses `HYPO_FOLLOWUP_GI` (20).

### Logged carbohydrate

The patient logs every carb event at its true onset step, but the grams and the glycaemic index are guesses. A meal logs the same `carb_estimate` that sizes its bolus, so the log and the dose agree; every other carb event — rescue, follow-up snack, trend carbs — draws its own:

    logged_grams   = grams * max(0, 1 + N(CARB_COUNT_UNDERBOLUS_BIAS, carb_count_error_sigma))
    gi_guess_sigma = GI_GUESS_SIGMA_BASE * (1.3 - s3)^2                  per patient
    logged_gi      = clip(GI + N(0, gi_guess_sigma), 0, 100)             per event

The logged curve is a second gamma, `gamma_curve(logged_grams, gi_gamma_params(logged_gi))`, laid at the true onset index and accumulated apart from the true one. A 0 g guess logs nothing — no curve and no event row. Nothing physiological reads the logged curve: `glucose_in`, the glycogen refill, postprandial insulin resistance, carbs-on-board and the delayed-HGO rebound all read the true one. The guesses are drawn from a generator keyed on the simulator seed alone, so a seed's BG trace is independent of them.

The step dict exports the logged curve as `total_carb` and the true curve as `total_carb_true`; the `carb_g` and `carb_gi` point-event channels carry the guess.


## Insulin Action Curves

Both curves are glucose-lowering action; the parameter table is SPEC/invariants.md §5.

Bolus (rapid-acting): gamma curve whose duration and theta scale with dose about a 5U reference, per analogue class in `BOLUS_VARIANTS` (rapid: aspart, lispro; ultra-rapid: faster aspart, ultra-rapid lispro). Helper: `bolus_pk_for_dose(dose, k, theta, dia) -> (k, theta, duration_minutes)`. Meal boluses scale `k` and `theta` by `1 + N(0, 0.05)` each.

    sqrt_excess = sqrt(dose) - sqrt(5)
    duration_h  = clip(BOLUS_DIA_BASE_HOURS + BOLUS_DIA_DOSE_SCALE * sqrt_excess,
                       BOLUS_DIA_MIN_HOURS, BOLUS_DIA_MAX_HOURS)
    theta       = gamma_theta * (1 + BOLUS_THETA_DOSE_SLOPE * sqrt_excess)
    k           = gamma_k

Basal (long-acting): Bateman curve from `basal_curve()`, per analogue in `BASAL_VARIANTS` (glargine U100, glargine U300, degludec):

    f(t)   = exp(-ke · t) − exp(-ka · t)          t in hours, sampled at each step's start
    tmax   = ln(ka / ke) / (ka − ke)               5.3 h, 12 h, 12 h
    t_half = ln 2 / ke                             13.9 h, 18.4 h, 25 h

The action window (`action_hours`: 73, 101, 133 h) ends where 3% of the untruncated area remains, so daily doses overlap as at steady state. A smootherstep over the first `BASAL_ONSET_RAMP_HOURS` (3 h) eases each dose in from zero, so the summed basal has no corner at an injection, and one over the last `CURVE_TAIL_CLIP_FRACTION` (1/6) of the window tapers the residual to zero. Normalized so the area equals the dose.

### Injection site quality (lipohypertrophy)

Every dose (basal, meal bolus, hyper correction, trend correction) is multiplied by a per-dose factor. Low `lifestyle_consistency` (s4) means poor site rotation and higher dose-to-dose variance; the PK shape (k, theta, duration) is set by the *intended* dose, only the absorbed amount varies:

    site_quality   ~ N(1.0, SITE_QUALITY_SIGMA_BASE * (1.5 - s4) ** 1.8)
    site_quality   = clip(site_quality, SITE_QUALITY_MIN, SITE_QUALITY_MAX)
    delivered_dose = intended_dose * site_quality

A basal dose contracts its factor toward 1.0 by `BASAL_SITE_QUALITY_DAMPING`, since long-acting depots absorb more consistently than rapid-acting ones.

### Logged insulin

The patient records the dose they injected, not the dose that reached the blood. Each insulin curve is accompanied by a logged curve scaled back to the intended amount, `delivered_curve * (intended_dose / delivered_dose)`, and by an event row carrying `intended_dose`. Only the delivered curve enters `glucose_out` and the HGO Hill function. The step dict exports the log as `total_insulin`, `basal_insulin` and `bolus_insulin`, the delivered amounts as `total_insulin_true`, `basal_insulin_true` and `bolus_insulin_true`, and the `bolus_u` / `basal_u` point events carry the intended dose.


## Insulin Sensitivity

Diurnal pattern (`phase_shift` and `daily_drift` smooth-step-blend across midnight from yesterday to today over `IS_DRIFT_TRANSITION_HOURS`):

    morning = IS_MORNING_AMPLITUDE * exp(-0.5 * ((hour - IS_MORNING_PEAK_HOUR - phase_shift) / 2.0)^2)
    night   = -IS_NIGHT_DIP_AMPLITUDE * exp(-0.5 * ((night_hour - IS_NIGHT_DIP_HOUR) / 2.0)^2)
    diurnal = 1.0 + morning + night

    IS(t)   = IS_base * diurnal * (1 + daily_drift) * illness_factor
              * stress_envelope
              * glucotox_factor * postprandial_ir
              * (1 + fast_noise)

| Term | Sampling | Cadence |
| --- | --- | --- |
| `daily_drift` | `N(0, IS_DAILY_DRIFT_SIGMA * (1.5 - s4))` — consistent-lifestyle patients swing less day-to-day | once per day, blended across midnight |
| `phase_shift` | `N(0, IS_DAWN_PHASE_DAILY_SIGMA)` | once per day, blended across midnight |
| `fast_noise` | AR(1) at `NOISE_AR1_RHO_METABOLIC` about `N(0, IS_FAST_NOISE_SIGMA)`; that sigma is 0, so the term is identically 0 | every step |
| `illness_factor` | ramps toward `illness_is_target` at `ILLNESS_IS_RAMP_RATE` per day; rests at 1.0 when healthy | always applied |
| `stress_envelope` | trapezoidal `envelope_intensity()` blending the raw factor against 1.0 | per event |

### Glucotoxicity

A slow EMA of true BG (3h half-life) drives transient insulin resistance when chronically elevated, closing a positive feedback loop (high BG → more IR → harder to bring down):

    glucotox_bg_ema += α * (BG - glucotox_bg_ema)   where α = 1 - 0.5^(dt / half_life)
    if glucotox_bg_ema > GLUCOTOX_BG_THRESHOLD:
        intensity = min(1, (ema - threshold) / (max_bg - threshold))
        glucotox_factor = 1 + GLUCOTOX_MAX_IS_INCREASE * intensity
    else:
        glucotox_factor = 1.0

### Postprandial insulin resistance

In T1DM the incretin / GLP-1 axis is blunted and there is no endogenous insulin response, so the meal-time sensitivity boost non-diabetics get is absent; the absorbing-carb state is if anything mildly insulin-*resistant*. IR is therefore raised while carbs absorb, saturating in active carb load:

    penalty = POSTPRANDIAL_IR_PENALTY_FACTOR * active_carb / (POSTPRANDIAL_IR_PENALTY_HALF + active_carb)
    postprandial_ir = 1 + penalty


## BG Delta Computation

    glucose_in  = total_carb_true + HGO
    glucose_out = total_insulin * ICR / IS(t)
    delta_BG    = BG_SCALE_FACTOR * (glucose_in - glucose_out)
    delta_BG   += Sg * (E(t) - BG)     # glucose-effectiveness restoring pull (see below)

Carbs enter as the true absorption curve; the logged curve the patient wrote down (see Logged carbohydrate) never appears here. `IS(t)` divides the insulin side only: insulin-resistant patients (IS > 1) clear less glucose per unit insulin. HGO suppression by insulin is handled separately by the Hill function (see Hepatic Glucose Output). Physiological guardrails, then the clamp:

    if BG > RENAL_THRESHOLD:
        delta_BG -= (BG - RENAL_THRESHOLD) * RENAL_CLEARANCE_RATE

    if BG < COUNTER_REGULATORY_THRESHOLD:
        delta_BG += COUNTER_REGULATORY_RATE * (COUNTER_REGULATORY_THRESHOLD - BG) / COUNTER_REGULATORY_THRESHOLD

    if BG < SEVERE_HYPO_THRESHOLD:
        severity = (SEVERE_HYPO_THRESHOLD - BG) / SEVERE_HYPO_THRESHOLD
        delta_BG += SEVERE_HYPO_GLUCAGON_RATE * severity

    BG(t+1) = clamp(BG(t) + delta_BG, BG_CLAMP_MIN, BG_CLAMP_MAX)

`BG_CLAMP_MIN` is 1 mg/dL. It is not a device floor — a real CGM stops reporting near 40 but the patient keeps falling, and clamping the dynamics at the reporting floor made a descent taper out there. 1 mg/dL is far below survivable, so it never binds physiologically; it exists to keep the Kovatchev log transform defined. The counter-regulatory and glucagon-dump terms plus the soft-bound headroom cap normally arrest a fall well above it.

### Glucose effectiveness (Bergman Sg) equilibrium

`Sg = glucose_effectiveness` is the per-patient Bergman minimal-model glucose effectiveness (a per-step reversion fraction), sampled lognormally around `GE_RATE = 0.068` and clipped to `[GE_RATE_MIN, GE_RATE_MAX] = [0.004, 0.150]` (~2–3× inter-individual spread; the floor prevents a pure integrator). `GE_RATE` is the MEDIAN of that lognormal, not its mean, so the median patient's reversion time constant is `1 / GE_RATE` ≈ 14.7 steps ≈ 74 min, which is what sets the 1–4 h BG autocorrelation. It is shorter than the 4 h bolus duration of action at the 5 U reference (`BOLUS_DIA_BASE_HOURS`, scaling as √dose), so the spring high-passes part of a bolus's sustained effect — raising `GE_RATE` further cancels more of it. The pull supplies the within-band mean reversion the renal / counter-regulatory guardrails do not: inside 70–180 net flux is otherwise integrated with an over-long autocorrelation.

`E(t)` is an Ornstein–Uhlenbeck process. With `rho = exp(-DT_MINUTES / (GE_EQ_TAU_HOURS * 60))`:

    mu = ge_anchor + ge_dawn_amplitude * ge_diurnal_profile(hour)
    E  = mu + rho * (E_prev - mu) + sqrt(1 - rho^2) * GE_EQ_SIGMA * ge_sigma_mult * N(0, 1)
    E  = max(E, GE_EQ_FLOOR)

The `sqrt(1 - rho^2)` factor makes the stationary std equal `GE_EQ_SIGMA * ge_sigma_mult`. `E`'s own timescale, not the strength of the Sg pull, is what keeps the 8h ACF near zero: `E` wanders enough to supply the distributional spread but decorrelates within hours, decoupling spread from the autocorrelation. Sg itself is deliberately weak, because a strong spring high-passes any input slower than its own time constant — insulin included. `GE_EQ_FLOOR = 40` sits above `SEVERE_HYPO_THRESHOLD = 25`, so the pull is always upward in a severe low (it aids, never opposes, the rescue). `ge_diurnal_profile(hour)` is a mean-zero wrapped-Gaussian dawn-phenomenon rhythm peaking at `GE_DAWN_PEAK_HOUR = 9` with width `GE_DAWN_WIDTH_HOURS = 5.5`, mean-subtracted over the 24h day so it adds rhythm without shifting the pooled mean; its per-patient amplitude `ge_dawn_amplitude` scales with the same dawn trait as the HGO surge.

Per-patient heterogeneity, sampled once in `generate_patient`:

    ir            = clip(exp(N(0, IR_LOGNORMAL_SIGMA)), IR_FACTOR_MIN, IR_FACTOR_MAX)
    ge_anchor     = clip(N(GE_EQ_ANCHOR_MEAN + GE_ANCHOR_IR_COUPLING * (ir - 1), GE_EQ_ANCHOR_SIGMA), 110, 210)
    ge_sigma_mult = clip(exp(N(0, GE_SIGMA_REL_SIGMA)), GE_SIGMA_MULT_CLIP)

| Constant | Value | Role |
| --- | --- | --- |
| `IR_LOGNORMAL_SIGMA` | 0.26 | lognormal sigma of `ir` |
| `IR_FACTOR_MIN` / `IR_FACTOR_MAX` | 0.4 / 2.0 | clip on `ir` |
| `GE_EQ_ANCHOR_MEAN` | 109 | population-mean anchor (mg/dL), before the clip to [110, 210] |
| `GE_ANCHOR_IR_COUPLING` | 30 | mg/dL per unit `ir - 1` |
| `GE_EQ_ANCHOR_SIGMA` | 15 | between-patient anchor spread |
| `GE_SIGMA_REL_SIGMA` | 0.16 | lognormal sigma of `ge_sigma_mult` |
| `GE_SIGMA_MULT_CLIP` | (0.68, 1.38) | clip on `ge_sigma_mult` |

The anchor's between-patient spread carries the per-patient mean-glucose heterogeneity, the IR coupling raising it for resistant patients on the high side that `GE_EQ_FLOOR` does not compress. `ge_sigma_mult` makes patients differ in *within*-patient variability rather than sharing one global `GE_EQ_SIGMA`. The same `ir` also seeds `is_base`, `icr`, and `correction_factor`.


## CGM Observation Model

The sensor reports a delayed-and-smoothed interstitial value with time-correlated multiplicative noise — never the instantaneous true BG. A first-order interstitial lag (Rebrin/Steil) is applied first, then AR(1) sensor noise multiplicatively. The timescale is **per patient**, `cgm_lag_minutes ~ clip(N(CGM_LAG_MEAN_MINUTES, CGM_LAG_SIGMA_MINUTES), *CGM_LAG_CLIP)` — 8 ± 4 min clipped to [0, 20]. The mean sits below the raw physiological 5–15 min because CGM firmware compensates much of the apparent lag, and the spread covers sensor generations from fully compensated to not at all, so a model trained here is lag-robust rather than tuned to one device:

    if cgm_lag_minutes > 0:                                  # 0 is inside the clip
        alpha_lag = 1 - exp(-DT_MINUTES / cgm_lag_minutes)
        IG       += alpha_lag * (BG_true - IG)
    else:
        IG        = BG_true                                  # fully compensated sensor
    ar_cgm      = NOISE_AR1_RHO_SENSOR * ar_cgm + NOISE_AR1_INNOV_SENSOR * N(0, CGM_NOISE_FRACTION)
    BG_observed = clip(IG * (1 + ar_cgm), BG_CLAMP_MIN, BG_CLAMP_MAX)

Multiplying the reading makes the noise std scale with BG, matching real CGM MARD characteristics. `NOISE_AR1_RHO_SENSOR = 0.92` (~42 min half-life, with `NOISE_AR1_INNOV_SENSOR = sqrt(1 - 0.92^2)`) gives smoothly-drifting offsets over 30-60 min windows rather than white-noise spikes. Every consumer of `BG_observed` — corrections, hypo detection, the exported CGM channel — sees this value, not the current step's true BG.


## Hepatic Glucose Output

Insulin-suppressed via a Hill function on EMA-smoothed insulin (proxies plasma insulin lag behind subcutaneous absorption, ~12 min half-life at `HGO_INSULIN_SMOOTHING_ALPHA = 0.25`):

    smoothed_ins  = α * insulin_per_step + (1-α) * smoothed_ins_prev
    suppression   = 1 / (1 + smoothed_ins / HGO_INSULIN_HALF_MAX)
    HGO_rate      = HGO_SUPPRESSED_FLOOR + (HGO_UNSUPPRESSED - HGO_SUPPRESSED_FLOOR) * suppression
    HGO_baseline  = max(0, HGO_rate * (1 + N(0, HGO_NOISE_SIGMA)) * (body_weight_kg / BODY_WEIGHT_MEAN_KG) * (DT_MINUTES / 60)
                            + (dawn_g_per_hr - night_dip_g_per_hr) * (DT_MINUTES / 60)) * glycogen_gate * alcohol_factor
    meal_rebound  = sum over active meal_hgo_effects of (magnitude * envelope_intensity) * (DT_MINUTES / 60)
    HGO(t)        = HGO_baseline + meal_rebound

- `HGO_INSULIN_HALF_MAX` is tuned so the anchor basal level (0.0516 U/step, the per-step basal of an ICR-8 patient) yields 4.95 g/hr = `HGO_BASE_GRAMS_PER_HOUR`, the balanced reference rate. `ICR_MEAN` is 5.6, so the population's typical basal per step sits above that anchor and its HGO a little below the reference. This preserves basal sizing — `ideal_basal = HGO_BASE_GRAMS_PER_HOUR * 24 * (body_weight_kg / BODY_WEIGHT_MEAN_KG) * is_base / ICR` gives near-zero net delta. The weight factor mirrors the per-step HGO scaling; `is_base` keeps the invariant across baseline insulin needs.
- At zero insulin, HGO climbs toward `HGO_UNSUPPRESSED_GRAMS_PER_HOUR` (DKA-like).
- `(dawn_g_per_hr - night_dip_g_per_hr)` is a cortisol-driven dawn surge (Gaussian peaking at `DAWN_HGO_PEAK_HOUR`) minus a deep-sleep trough (Gaussian at `NIGHT_HGO_DIP_HOUR`), added in g/hr rather than as a multiplier so the Hill suppression does not cancel it — this produces the dawn phenomenon.
- `alcohol_factor` (trapezoidal envelope around 1.0) suppresses HGO on top of insulin's suppression; `glycogen_gate` ramps HGO down when the reservoir is depleted.
- `meal_rebound` is additive, not multiplicative. Helper: `compute_hgo_rate(insulin_per_step) -> g/hr`.

### Delayed-meal HGO rebound

Each meal above `DELAYED_HGO_MEAL_THRESHOLD_GRAMS` schedules a positive HGO bump 3.5-5.5h later lasting 2.7-7.0h, shaped by the trapezoidal `envelope_intensity()` with `DELAYED_HGO_RAMP_HOURS` ramps. Models the delayed gluconeogenesis from amino acids and cortisol response that drive nocturnal hyperglycemia after a large dinner:

    excess     = carb_amount - DELAYED_HGO_MEAL_THRESHOLD_GRAMS
    magnitude  = min(DELAYED_HGO_MAX_BUMP, DELAYED_HGO_PER_GRAM * excess)   (g/hr)
    delay      ~ U(DELAYED_HGO_DELAY_HOURS_MIN, DELAYED_HGO_DELAY_HOURS_MAX)
    duration   ~ U(DELAYED_HGO_DURATION_HOURS_MIN, DELAYED_HGO_DURATION_HOURS_MAX)

### Glycogen reservoir

Hepatic glycogen is a finite store gating the glycogenolysis-sourced fraction of HGO, drained and refilled each step. The refill is a "background" channel — not subtracted from BG-bound carbs, since ICR is empirically tuned to net BG response; the only coupling back to BG dynamics is `glycogen_gate` reducing future HGO:

    if glycogen < GLYCOGEN_CAPACITY * GLYCOGEN_LOW_THRESHOLD_FRACTION:
        availability  = glycogen / (GLYCOGEN_CAPACITY * GLYCOGEN_LOW_THRESHOLD_FRACTION)
        glycogen_gate = (1 - GLYCOGEN_DRAIN_FRACTION) + GLYCOGEN_DRAIN_FRACTION * availability
    else:
        glycogen_gate = 1.0

    glycogen -= HGO(t) * GLYCOGEN_DRAIN_FRACTION        (drain from glycogenolysis)
    glycogen += total_carb_true * GLYCOGEN_REFILL_FRACTION   (refill from absorbed carbs)
    glycogen  = clip(glycogen, 0, GLYCOGEN_CAPACITY)


## Correction Behavior

Hypo correction (BG_observed < hypo_threshold):

    skill_avg         = (attentiveness + dosing_competence) / 2
    dev               = skill_avg - HYPO_THRESHOLD_SKILL_MID
    gain              = HYPO_THRESHOLD_SKILL_SPAN / (HYPO_THRESHOLD_SKILL_MID - SKILL_MIN)   if dev < 0
                        HYPO_THRESHOLD_SKILL_SPAN / (SKILL_MAX - HYPO_THRESHOLD_SKILL_MID)   otherwise
    hypo_threshold    = HYPO_THRESHOLD_MEDIAN + gain * dev

    rescue_cob        = sum of the not-yet-absorbed correction-carb curve from t onward (g)
    awareness         = COB_AWARENESS_BASE + COB_AWARENESS_SKILL * dosing_competence
    iob               = sum of the not-yet-acted bolus curve from t onward (U)
    cob_consideration = awareness * (rescue_cob * BG_SCALE_FACTOR
                                     - HYPO_RESCUE_IOB_WEIGHT * iob * correction_factor)
    projected_bg      = BG_observed + cob_consideration
    deficit           = hypo_threshold + HYPO_RESCUE_TARGET_MARGIN - projected_bg
    correction_grams  = max(HYPO_RESCUE_MIN_GRAMS, deficit / BG_SCALE_FACTOR
                            * (HYPO_RESCUE_DEFICIT_GAIN + panic_factor * HYPO_RESCUE_PANIC_GAIN))

The trigger is the per-patient `hypo_threshold`, sampled once in `generate_patient` from that patient's own `skill_avg`: attentive/competent patients act on the drop earlier. The gain is piecewise about `HYPO_THRESHOLD_SKILL_MID = 0.5` because `SKILL_MIN` and `SKILL_MAX` (0.15 / 0.98) are not symmetric around it, so the population spans `HYPO_THRESHOLD_MEDIAN ± HYPO_THRESHOLD_SKILL_SPAN` — 58 to 78 mg/dL, median 68 — with median skill landing on the median threshold. The same value defines "low" everywhere else the patient acts on it. A meal bolus is gated on the patient glancing at the CGM first — probability `BOLUS_BG_CHECK_BASE_PROB + 0.05 * attentiveness`, so 0.95 to 0.999 and not a certainty — and on that glance the bolus is skipped below the threshold and scaled by `BOLUS_REDUCE_FACTOR_BASE + 0.3 * dosing_competence` within `BOLUS_REDUCE_MARGIN` (15 mg/dL) above it.

The deficit is measured against `projected_bg`, not the current reading: this is the recheck half of the rule of 15, so rescue glucose already swallowed and still absorbing is counted before more is eaten, and half the drop the bolus insulin still acting will cause is subtracted. One rescue covers the whole projected deficit, so a deep low is not treated in small, frequent doses. Only correction carbs are counted (tracked separately from meal carbs, which arrive with a bolus attached), `BG_SCALE_FACTOR` converts grams to the mg/dL they will raise, and `awareness` = `COB_AWARENESS_BASE + COB_AWARENESS_SKILL * dosing_competence` rises to 0.89 at the `SKILL_MAX` ceiling of 0.98, approaching but never reaching its 0.90 limit. When `projected_bg >= hypo_threshold` the patient waits instead of eating. With probability `RAGE_EAT_PROBABILITY_BASE * (1 - dosing_competence)` the arithmetic is discarded (`cob_consideration = 0`) and the patient treats on the raw reading.

Severe hypo (BG_observed < `SEVERE_HYPO_THRESHOLD`) bypasses the CGM check interval, but a `SEVERE_HYPO_REFRACTORY_MIN` gate still applies between back-to-back doses so stacked carbs don't sawtooth into post-correction hypers. After any hypo correction, basal is scaled down along a sin² envelope, to `POST_HYPO_BASAL_SUSPEND_FACTOR` at its deepest, over `POST_HYPO_BASAL_SUSPEND_DURATION_HOURS` (pump-suspend / temp-basal analogue); overlapping envelopes combine as `1 − Π(1 − envelope)`, and skill-gated patients (`skill_avg > HYPO_FOLLOWUP_SKILL_THRESHOLD`) eat a slow-carb follow-up snack of `HYPO_FOLLOWUP_FRACTION × correction_grams`.

Hyper correction (BG_observed > eff_high_thresh):

    eff_high_thresh   = BG_HIGH_THRESHOLD - 25 * skill_avg
    iob_consideration = IOB * correction_factor * (0.7 + 0.3 * dosing_competence)
    adjusted_excess   = max(0, (BG_observed - BG_TARGET) - iob_consideration)
    correction_dose   = max(0.5, adjusted_excess / correction_factor * (1 + noise))
    urgency           = min(3, 1 + max(0, (BG_observed - BG_HIGH_THRESHOLD) / 50))
    patience          = patience_time / urgency

Subtracting the insulin-on-board term before sizing stops patients from stacking corrections. Urgency saturates at 3 (reached at BG = 300), shortening the patience window for sustained highs.

Above `RAGE_BOLUS_BG_THRESHOLD` rage bolusing may occur, with probability proportional to `(1.2 - dosing_competence)`.


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

The asymmetric thresholds (150 / 115) intentionally bias toward correcting persistent hyperglycemia faster than persistent mild hypoglycemia. On the high path skill scales only partially and `extreme_boost` accelerates recovery when the 3-day or single-day mean runs above 200; the low path keeps full skill scaling.


## Behavioral & Stochastic Features

Mechanisms that perturb the deterministic core above, closing the gap between an idealized model and a free-living patient.

### Soft-bound BG headroom cap

Ahead of the hard clamp, each step's `delta_BG` is capped to a fraction of the headroom remaining to the soft bound, so the dynamics asymptote toward the bounds instead of slamming into the clamp:

    projected = BG(t) + delta_BG
    if projected < BG_SOFT_FLOOR:
        headroom = max(0, BG(t) - BG_CLAMP_MIN)
        delta_BG = max(delta_BG, -SOFT_APPROACH_FRACTION * headroom)
    if projected > BG_SOFT_CEILING:
        headroom = max(0, BG_CLAMP_MAX - BG(t))
        delta_BG = min(delta_BG,  SOFT_APPROACH_FRACTION * headroom)

The hard clamp still runs as a backstop after this cap. `BG_SOFT_FLOOR` is 0 mg/dL and `BG_SOFT_CEILING` 500, both outside the hard `[BG_CLAMP_MIN, BG_CLAMP_MAX] = [1, 400]` bounds, so the cap engages only for a step that would overshoot the clamp outright: the ceiling cap needs a single step past 500 and the floor cap one past 0. Everywhere inside the bounds the raw delta passes through undamped.

### Per-step absorption noise

An AR(1) multiplicative term on the per-step contributions read from the accumulation arrays, modelling moment-to-moment absorption variation the smooth gamma curves cannot capture. Active only when the underlying curve is non-zero, and both sigmas are currently 0, so the term is identically 0:

    total_carb_true *= max(0, 1 + ar_carb)      if total_carb_true > 0    ar_carb ~ AR(1) of N(0, CARB_ABSORPTION_NOISE_SIGMA)
    total_insulin   *= max(0, 1 + ar_insulin)   if total_insulin > 0      ar_insulin ~ AR(1) of N(0, INSULIN_ABSORPTION_NOISE_SIGMA)

The logged carb curve is read straight from its accumulator; absorption noise is physiology, not a property of the record.

### Stress IS envelope

Stress events transiently raise IS (more resistant), with per-day probability `STRESS_PROBABILITY_BASE - STRESS_LIFESTYLE_WEIGHT * s4`:

    is_factor          ~ U(STRESS_IS_FACTOR_MIN, STRESS_IS_FACTOR_MAX)
    duration           ~ U(STRESS_DURATION_HOURS_MIN, STRESS_DURATION_HOURS_MAX)
    stress_envelope(t) = 1 + (is_factor - 1) * envelope_intensity(t; start, end, STRESS_IS_RAMP_HOURS)

### Alcohol HGO suppression envelope

Drinking suppresses HGO multiplicatively, with an onset delay, plateau, and ramp-down. This multiplies the Hill-derived HGO baseline, separately from insulin suppression, and accounts for the nocturnal-hypo pattern after evening drinking:

    hgo_reduction ~ U(ALCOHOL_HGO_REDUCTION_MIN, ALCOHOL_HGO_REDUCTION_MAX)
    duration      ~ U(ALCOHOL_DURATION_HOURS_MIN, ALCOHOL_DURATION_HOURS_MAX)
    onset_delay   ~ U(ALCOHOL_ONSET_DELAY_HOURS_MIN, ALCOHOL_ONSET_DELAY_HOURS_MAX)
    alcohol_factor(t) = 1 - hgo_reduction * envelope_intensity(t; start+onset, start+onset+duration,
                                                                ALCOHOL_HGO_RAMP_HOURS)

### Trend-based anticipatory corrections

Attentive patients with sufficient skill act on a recent BG trend before crossing a threshold. From a sliding window of the last `TREND_CORRECTION_WINDOW_STEPS` BG samples, a preemptive correction bolus is considered when `trend > TREND_HIGH_RATE_THRESHOLD` and BG is approaching the upper band; a preemptive snack when `trend < TREND_LOW_RATE_THRESHOLD` and BG is approaching the lower band. The projected rise/fall over the next `2 * TREND_CORRECTION_WINDOW_STEPS` steps sizes the dose / carbs.

    trend = (window[-1] - window[0]) / (TREND_CORRECTION_WINDOW_STEPS - 1)   (mg/dL/step)

### Anomalous absorption events

With per-day probability `ANOMALOUS_EVENT_PROBABILITY`, that day's first meal absorbs anomalously slowly — delayed gastric emptying or an unusual food:

    GI ~ U(0, ANOMALOUS_MEAL_GI_MAX)

### Rare event days

With per-day probability `RARE_EVENT_PROBABILITY`, three of the four skills — dietary discipline (s1), dosing competence (s3), lifestyle consistency (s4) — are degraded for that day; attentiveness (s2) is unchanged. Models the chaotic days (illness onset, travel, emotional events) even well-controlled patients have:

    skill_penalty ~ RARE_EVENT_SKILL_REDUCTION + U(0, 0.3)
    s_i(today)    = max(s_i - skill_penalty, 0.05)   for i in {1, 3, 4}


## Unit Conventions

All curve values are in "amount per step" units:

| Channel | Unit | Normalization |
| --- | --- | --- |
| Carb curves (true, and logged) | grams per step | sum of curve = total grams |
| Insulin curves | units per step | sum of curve = total units |
| HGO | grams per step | rate g/hr converted via `DT_MINUTES / 60` |

Both `gamma_curve` and `basal_curve` normalize so that `sum(values) = total_amount`; `gamma_curve` reaches that per-step meaning by integrating the gamma across each step (see Carbohydrate Absorption Curves). There is no `flat_curve` — `basal_curve` (Bateman PK, smooth onset/peak/decline) replaced it. Never pass a rate where `total_amount` is expected.
