"""
T1DM patient-behavior simulator: carb/insulin/sensitivity/exercise curves -> BG delta.
Seed-driven, step-based; call generate() to advance 5 minutes.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

# Time resolution
DT_MINUTES = 5  # Time step in minutes
STEPS_PER_DAY = 24 * 60 // DT_MINUTES  # 288 steps per day

# Basal PK settles ~36-48h; shorter warmup leaves cold-start ramp peaking ~280 mg/dL. Whole days.
SIMULATOR_WARMUP_HOURS = 48.0

# Skill correlation
SKILL_CORRELATION = 0.7  # Off-diagonal correlation in skill covariance matrix
SKILL_VARIANCE = 0.6  # [HIVAR 2x] 0.3→0.6 — wider skill spread
SKILL_MIN = 0.15  # [HIVAR 2x] 0.25→0.15 — low-skill floor
SKILL_MAX = 0.98  # [HIVAR 2x] 0.95→0.98 — high-skill ceiling

# Wake/sleep
WAKE_TIME_MEAN_HOURS = 7.5  # 7.5h wake -> breakfast ~08:00, diurnal peak hour 8 (Ohio/Shanghai).
WAKE_TIME_SIGMA_BASE = 0.5  # hours; late wakers otherwise eat inside the night bolus window
SLEEP_DURATION_MEAN_HOURS = 7.5
SLEEP_DURATION_SIGMA_HOURS = 2.0  # [HIVAR 2x] 1.0→2.0 — sleep-length scatter

# Meals
MEALS_BASE = 4  # 3->4: afternoon slot fills lunch->dinner sag vs Ohio; jitter keeps it aperiodic.
MEALS_EXTRA_LAMBDA = 1.5  # Poisson lambda*(1-s1) extra snacks per day
MEAL_TIME_OFFSETS_HOURS = [0.5, 5.0, 8.5, 12.0]  # hrs from wake: breakfast/lunch/snack/dinner.
MEAL_TIME_JITTER_BASE_MIN = 30.0  # small, so the night bolus window stays meal-free
MEAL_CARB_MEANS = [32.0, 42.0, 20.0, 35.0]  # g/meal, cut from [48,63,20,75]: overshot Ohio +50/+27.
MEAL_CARB_SIGMA = 16.0  # g/meal SD. 44->16: the max(0,.) floor inflated carbs/day ~40% over Ohio.
MEAL_CARB_DISCIPLINE_SCALE = 0.7  # How much s1 reduces carb intake
# Per-patient lognormal appetite multiplier (orthogonal to skill) supplies Ohio's right skew.
MEAL_APPETITE_LOG_SIGMA = 0.45  # lognormal sigma; tuned so pooled carbs/day sd ~90g matches Ohio.
MEAL_APPETITE_CLIP = (0.45, 2.05)  # clips lognormal tails to ~90-370+ g/day eaters.
MEAL_CARB_SCALE = 1.51  # meal carbs/day 189 g, the channel total before the protein/fat tail went.
SNACK_CARB_MEAN = 20.0
SNACK_CARB_SIGMA = 20.0  # [HIVAR 2x] 10.0→20.0 — unused by generation

# Carb curve peak = (k-1)*theta; FAST_CARB_K/THETA are test-only, production uses MIXED_MEAL ranges.
FAST_CARB_K = 3.0  # Gamma shape for fast carbs (peak ~40 min)
FAST_CARB_THETA = 20.0  # Gamma scale for fast carbs (minutes)
# slow_carb_preference ~0.18 (mid-skill) keeps envelope peak ~100min (Ohio); 0.55 -> past 200min.
SLOW_CARB_PREFERENCE_BASE = 0.10  # Base probability of choosing slow carbs
SLOW_CARB_PREFERENCE_SKILL_BONUS = 0.15  # Added probability from s1

# Hypo correction carbs (glucose tablets / juice — kick in faster than meal carbs)
HYPO_CARB_K = 2.0
HYPO_CARB_THETA = 15.0  # Peak ~15 min

# Carb curve noise
CARB_CURVE_K_NOISE = 0.2  # [HIVAR 2x] 0.1→0.2 — absorption-shape scatter
CARB_CURVE_THETA_NOISE = 0.2  # [HIVAR 2x] 0.1→0.2 — absorption-shape scatter

# Mixed-meal composition (each meal becomes 2-5 overlapping carb components)
MIXED_MEAL_MIN_COMPONENTS = 2
MIXED_MEAL_EXTRA_COMPONENTS_LAMBDA = 1.5  # Poisson, added to MIN
MIXED_MEAL_MAX_COMPONENTS = 5
MIXED_MEAL_DIRICHLET_ALPHA = 1.5  # Higher = more uniform fractions per component
MIXED_MEAL_FAST_K_RANGE = (2.0, 3.5)
MIXED_MEAL_FAST_THETA_RANGE = (7.5, 11.0)
MIXED_MEAL_MED_K_RANGE = (3.0, 4.5)
MIXED_MEAL_MED_THETA_RANGE = (10.0, 14.0)
# Component peaks (k-1)*theta: fast 8-28 min, medium 20-49 min, slow 28-70 min.
MIXED_MEAL_SLOW_K_RANGE = (3.5, 5.0)
MIXED_MEAL_SLOW_THETA_RANGE = (11.0, 17.5)
MIXED_MEAL_MED_WEIGHT_BASE = 0.4  # Base weight for medium-speed components

# Independent per-patient axes: body_weight_kg scales HGO/basal, IR factor scales ICR and is_base.
BODY_WEIGHT_MEAN_KG = 75.0
BODY_WEIGHT_SIGMA_KG = 36.0  # [HIVAR 2x] 18.0→36.0 — body-weight spread
BODY_WEIGHT_MIN_KG = 42.0  # [HIVAR 2x] 45.0→42.0 — light-weight floor
BODY_WEIGHT_MAX_KG = 150.0  # [HIVAR 2x] 130.0→150.0 — heavy-weight ceiling
IR_LOGNORMAL_SIGMA = 0.26  # widened IR spread; anchor sigma alone saturates at ~14 mg/dL sd.
IR_FACTOR_MIN = 0.4  # [HIVAR 2x] 0.5→0.4 — sensitive end
IR_FACTOR_MAX = 2.0  # [HIVAR 2x] 1.5→2.0 — resistant end
IR_TO_IS_NOISE_SIGMA = 0.2  # [HIVAR 2x] 0.1→0.2 — is_base decoupling
IR_TO_ICR_NOISE_SIGMA = 0.24  # [HIVAR 2x] 0.12→0.24 — ICR decoupling

# is_base derives from insulin_resistance_factor; IS_BASE_MEAN/SIGMA is the unused lognormal ref.
IS_BASE_MEAN = 1.0
IS_BASE_SIGMA = 0.2
IS_DAILY_DRIFT_SIGMA = 0.2  # [HIVAR 2x] 0.1→0.2 — day-to-day sensitivity drift
IS_FAST_NOISE_SIGMA = 0.0  # [MINNOISE] step-to-step insulin-resistance noise minimized to zero
IS_DAWN_PHASE_DAILY_SIGMA = 3.0  # [HIVAR 2x] 1.5→3.0 — dawn-timing scatter
IS_DRIFT_TRANSITION_HOURS = 4.0  # Smooth blend across midnight from prev to today's drift/phase

# Diurnal IS: dawn rise ~8am (cortisol+HGO), milder evening peak, most sensitive ~11pm-2am.
IS_MORNING_PEAK_HOUR = 7.5    # Morning resistance peak
IS_MORNING_AMPLITUDE = 0.05  # 0.30->0.05: 0.30 caused +37 mg/dL overshoot; real IR is 5-15%.
IS_EVENING_PEAK_HOUR = 20.0   # Evening resistance peak
IS_EVENING_AMPLITUDE = 0.0  # Zeroed: 0.08 slowed dinner clearance 8%; real cohorts show no bulge.
IS_NIGHT_DIP_HOUR = 2.0       # Nighttime sensitivity peak (low resistance)
IS_NIGHT_DIP_AMPLITUDE = 0.15 # How much more sensitive at night

# Illness
ILLNESS_PROBABILITY_BASE = 0.06  # Per-day probability of getting sick
ILLNESS_HEALTH_WEIGHT = 0.8  # How much s4 reduces illness probability
ILLNESS_RECOVERY_PROB = 0.2  # Geometric distribution parameter
ILLNESS_IS_FACTOR_MIN = 1.25  # [HIVAR 2x] 1.3→1.25 — illness IR range
ILLNESS_IS_FACTOR_MAX = 3.0  # [HIVAR 2x] 2.5→3.0 — illness IR range
ILLNESS_IS_RAMP_RATE = 0.4  # How fast illness IS factor changes per day (0 to 1)

# Basal insulin (long-acting); ideal_basal is derived from HGO and ICR in generate_patient().
BASAL_DOSE_SIGMA = 9.0  # [HIVAR 2x] 4.5→9.0 — inter-patient basal-dose scatter
BASAL_DOSE_COMPETENCE_NOISE = 0.1  # 0.15->0.1: real basal CV is 5-15% (glargine~12%, degludec~6%).
BASAL_DURATION_HOURS = 28.0  # Test ref; per-patient uses BASAL_VARIANTS (26h/42h), 24h cadence.
BASAL_DURATION_HOURS_MIN = 15.0  # Legacy basal-duration span bound — no longer read
BASAL_DURATION_HOURS_MAX = 34.0  # Legacy basal-duration span bound — no longer read
BASAL_MISS_PROB_BASE = 0.02  # 0.10->0.02: low-skill missed ~35% at 0.10; real MDI skip <3%.
BASAL_MISS_SKILL_SCALE = 5.0  # How much skills reduce miss probability
BASAL_KA_PER_HOUR = 0.30  # tmax=ln(ka/ke)/(ka-ke)~6.3h; peak between glargine(~4h)/degludec(~9h).
BASAL_KE_PER_HOUR = 0.07  # Half-life ~9.9h; dose stays within ~50% of peak across cadence window.
BASAL_TAIL_CLIP_HOURS = 5.0  # Smootherstep tapers residual so consecutive doses join, no tail-step.
# Unused: dose duration=basal_duration_hours (26h/42h) exceeds 24h cadence, tails overlap overnight.
BASAL_PK_OVERLAP_FRACTION = 1.00

# Contracts lipohypertrophy multiplier toward 1.0: basal sites absorb more consistently than bolus.
BASAL_SITE_QUALITY_DAMPING = 0.30
# Legacy aliases for tests/warmup math; with Bateman PK read as time-to-peak / tail-clip, not ramps.
BASAL_RAMP_UP_HOURS = 4.0
BASAL_RAMP_DOWN_HOURS = BASAL_TAIL_CLIP_HOURS

# Bolus/basal PK/PD sourced from prescribing info; curve is glucose-lowering action, not plasma.

# Bolus analogues -> gamma_curve: peak=(k-1)*theta min, dia_base_hours=DIA at the 5U reference dose.

# aspart (NovoLog): onset ~0.25h, peak ~1h, DIA ~4h. lispro: onset 0.25-0.5h, peak 1-2h, DIA ~5h.
BOLUS_VARIANTS = {
    "aspart": {"gamma_k": 3.0, "gamma_theta": 30.0, "dia_base_hours": 4.0},  # peak ~60 min
    "lispro": {"gamma_k": 3.0, "gamma_theta": 37.5, "dia_base_hours": 5.0},  # peak ~75 min
}

# Default bolus PK is aspart's; duration scales BASE+SCALE*(sqrt(dose)-sqrt(5)).
BOLUS_GAMMA_K = BOLUS_VARIANTS["aspart"]["gamma_k"]
BOLUS_GAMMA_THETA = BOLUS_VARIANTS["aspart"]["gamma_theta"]
BOLUS_DIA_BASE_HOURS = BOLUS_VARIANTS["aspart"]["dia_base_hours"]
BOLUS_DURATION_HOURS = 4.0  # Legacy typical duration; new code uses bolus_pk_for_dose()
BOLUS_DIA_DOSE_SCALE = 0.6  # Hours added per unit of sqrt(dose) - sqrt(5)
BOLUS_DIA_MIN_HOURS = 2.0
BOLUS_DIA_MAX_HOURS = 7.5
BOLUS_THETA_DOSE_SLOPE = 0.06  # Theta multiplier per unit of sqrt(dose) - sqrt(5)
ICR_MEAN = 8.0   # 11.0->8.0: raises TDD to clinical 0.5-0.7 U/kg/day, basal:bolus ~50/50 (AZT1D).
ICR_SIGMA = 2.0
# Bolus policy: count, clock time and dose are drawn independent of meals, carbs and BG.
BOLUS_NIGHT_START_HOUR = 1.0  # clock hour the meal-free night window opens
BOLUS_NIGHT_HOURS = 4.0  # night window length; day events fall anywhere outside it
BOLUS_NIGHT_EVENTS_PER_DAY = 16.0  # Poisson rate of bolus events in the night window
BOLUS_DAY_EVENTS_PER_DAY = 8.0  # Poisson rate of bolus events outside it
BOLUS_NIGHT_UNIT_SHARE = 0.35  # share of daily bolus units given in the night window
BOLUS_DOSE_LOG_SIGMA = 0.3  # lognormal sigma of one bolus dose around its window's median
BOLUS_BALANCE_GAIN = 1.1  # a day's bolus units over the units that clear its planned glucose

# Basal analogues -> basal_curve (Bateman PK): f(t)=exp(-ke t)-exp(-ka t), tmax=ln(ka/ke)/(ka-ke).

# glargine: half-life ~12h, tmax~6.8h, action 26h. degludec: half-life >25h, tmax~11.5h, action 42h.
BASAL_VARIANTS = {
    "glargine": {"ka": 0.30, "ke": 0.058, "action_hours": 26.0, "tail_clip_hours": 4.0},
    "degludec": {"ka": 0.20, "ke": 0.028, "action_hours": 42.0, "tail_clip_hours": 6.0},
}
BASAL_DOSE_INTERVAL_HOURS = 24.0  # Both analogues are injected once daily.

# Insulin stacking
CGM_CHECK_INTERVAL_ATTENTIVE = 20  # Minutes between checks for attentive patient
CGM_CHECK_INTERVAL_INATTENTIVE = 240  # Minutes for inattentive patient
CORRECTION_HORIZON_STEPS = 48  # a correction factor is the true-BG drop this long after 1 U
BG_LOW_THRESHOLD = 60.0  # Hypo-band floor label; the acting trigger is per-patient hypo_threshold.
# Per-patient hypo threshold centred on HYPO_THRESHOLD_MEDIAN; drives the rescue AND blocks boluses.
HYPO_THRESHOLD_MEDIAN = 80.0      # threshold of a median-skill patient (mg/dL)
HYPO_THRESHOLD_SKILL_SPAN = 10.0  # mg/dL dev at each skill end; population spans MEDIAN +/- this.
HYPO_THRESHOLD_SKILL_MID = 0.5  # median skill_avg; piecewise since SKILL_MIN/MAX aren't symmetric.

# Pre-meal BG-awareness: a scheduled bolus otherwise dumps into an already-hypo patient (sawtooth).

# Skip threshold is per-patient hypo_threshold, not global 75 (mismatched cautious/stoic patients).
BOLUS_REDUCE_MARGIN = 30.0  # Band above hypo_threshold where bolus reduces, not skips (was 75-105).
BOLUS_REDUCE_FACTOR_BASE = 0.3  # Reduction floor*(1+0.3*dosing_competence); near-low bolus halved+.
BOLUS_BG_CHECK_BASE_PROB = 0.95  # Prob of pre-bolus CGM check, +0.05*attentive; nearly all gated.

# Hypo correction
HYPO_RESCUE_TARGET_MARGIN = 20.0  # mg/dL above the patient's threshold a rescue aims for
HYPO_RESCUE_MIN_GRAMS = 4.0  # smallest rescue, g
HYPO_RESCUE_DEFICIT_GAIN = 1.0  # grams per gram needed to close the deficit at BG_SCALE_FACTOR
HYPO_RESCUE_PANIC_GAIN = 0.5  # extra gain per unit panic_factor
HYPO_PANIC_FACTOR_BASE = 1.0  # How much extra is eaten, scaled by 1/s3

# Exercise
EXERCISE_PROBABILITY_BASE = 0.3  # Base daily probability
EXERCISE_SKILL_BONUS = 0.4  # Added probability from s4
EXERCISE_TIME_MEAN_OFFSET_HOURS = 11.0  # wake+11h ~18:30; shifted with dinner move to 19:30.
EXERCISE_TIME_SIGMA_HOURS = 4.0  # [HIVAR] 2→4 — wider exercise-timing scatter.
EXERCISE_DURATION_MEAN_MIN = 75.0  # Ohio mean ~86min sd~57; patient samples own mean ~N(75,45).
EXERCISE_DURATION_MEAN_SIGMA_MIN = 90.0  # 45->90: wider across-patient session-length spread.
EXERCISE_DURATION_MEAN_MIN_CLAMP = (12.0, 225.0)  # (15,200)->(12,225): wider session-length clamp.
EXERCISE_DURATION_SIGMA_MIN = 40.0  # [HIVAR 2x] 20.0→40.0 — within-patient session-length noise
EXERCISE_CARB_EQUIV_PER_MIN = 0.5  # Negative carb equivalent per minute of exercise
# Exercise blocked below hypo_threshold+margin (negative food); bar is higher than for eating.
EXERCISE_HYPO_MARGIN = 20.0
EXERCISE_GAMMA_K = 3.0
EXERCISE_GAMMA_THETA = 15.0

# HGO: UNSUPPRESSED at zero insulin, saturates to SUPPRESSED floor; HALF_MAX tunes basal to ~9 g/hr.
HGO_BASE_GRAMS_PER_HOUR = 6.0  # g/h the basal dose is sized to cancel; under Hill HGO at basal
HGO_UNSUPPRESSED_GRAMS_PER_HOUR = 18.0  # Rate with no insulin (DKA-like)
HGO_SUPPRESSED_FLOOR_GRAMS_PER_HOUR = 6.0  # Maximum suppression
HGO_INSULIN_HALF_MAX = 0.0198  # 0.025->0.0198: h=B*(B-6)/(96*(18-B)), B=8.25 -> Hill(0.086)=8.25.
HGO_NOISE_SIGMA = 0.0  # [MINNOISE] step-to-step hepatic-output noise minimized to zero
HGO_INSULIN_SMOOTHING_ALPHA = 0.25  # EMA factor for the insulin level fed into the Hill function.
# Models plasma-insulin lag (~10-15min) behind SC absorption; half-life ~12min at alpha=0.25.

# Circadian HGO: dawn cortisol surge ~6-7am, deep-sleep dip ~2-3am; applied after Hill suppression.

# Per-patient dawn/night amplitude sampled in generate_patient (dawn/night_hgo_*_amplitude).
DAWN_HGO_PEAK_HOUR = 3.0  # clock hour of the dawn HGO surge; offsets the night bolus window
DAWN_HGO_SIGMA_HOURS = 2.5  # 3.5->2.5: spans midnight-noon, sustains the midnight-6am rise (Ohio).
DAWN_HGO_AMPLITUDE_MEAN = 10.0  # g/h at the surge peak, per-patient mean
DAWN_HGO_AMPLITUDE_SIGMA = 4.0  # [HIVAR 2x] 2.0→4.0 — dawn-surge spread
NIGHT_HGO_DIP_HOUR = 2.0             # Hour of deep-sleep HGO trough
NIGHT_HGO_DIP_SIGMA_HOURS = 2.5      # Narrower so it ends before dawn surge starts
NIGHT_HGO_DIP_AMPLITUDE_MEAN = 3.5  # 0.3->3.5: trough pulls BG below day (Ohio); was inverted.
NIGHT_HGO_DIP_AMPLITUDE_SIGMA = 0.5  # [HIVAR 2x] 0.25→0.5 — night-dip spread
# Net daily HGO integral ~+36 g/day (dawn ~41, dip ~4); without a strong surge dawn doesn't appear.

# Glycogen reservoir drains under HGO, refills from carbs; depleted -> floors at gluconeogenesis.
GLYCOGEN_CAPACITY_GRAMS = 100.0  # Maximum hepatic glycogen
GLYCOGEN_INITIAL_FRACTION = 0.7  # Patients start moderately full
GLYCOGEN_DRAIN_FRACTION = 0.35  # 0.5->0.35: at 0.5, depletion halved the dawn surge at hour 03-05.
GLYCOGEN_REFILL_FRACTION = 0.20  # Fraction of absorbed carbs stored as glycogen
GLYCOGEN_LOW_THRESHOLD_FRACTION = 0.15  # Below this fraction of capacity, HGO ramps down

# Glucotoxicity: slow BG EMA drives added IR, closing a positive feedback loop (high BG -> more IR).
GLUCOTOX_BG_EMA_HALF_LIFE_HOURS = 3.0  # Short; longer memory left a 4-12h ACF real CGM lacks.
GLUCOTOX_BG_THRESHOLD = 200.0  # Above this EMA value, IS starts to climb
GLUCOTOX_BG_FOR_MAX = 350.0  # EMA value at which the maximum IR multiplier is applied
GLUCOTOX_MAX_IS_INCREASE = 0.15  # Up to 15% more resistant at saturating BG

# Postprandial IR: T1DM lacks the GLP-1 boost; absorbing-carb state is mildly IR instead (~4% cap).
POSTPRANDIAL_IR_PENALTY_FACTOR = 0.04
POSTPRANDIAL_IR_PENALTY_HALF = 1.5  # g/step active carb at half-max penalty

# Injection-site quality: per-dose multiplier; sigma scales inversely with lifestyle_consistency.
SITE_QUALITY_SIGMA_BASE = 0.2  # [HIVAR 2x] 0.1→0.2 — per-dose absorption scatter
SITE_QUALITY_MIN = 0.35  # [HIVAR 2x] 0.5→0.35 — malabsorption floor
SITE_QUALITY_MAX = 1.65  # [HIVAR 2x] 1.4→1.65 — absorption-surge ceiling

# Delayed HGO rebound: large meals bump HGO 3.5-5.5h later; drives highs after a big dinner.
DELAYED_HGO_MEAL_THRESHOLD_GRAMS = 60.0  # Meals above this trigger a rebound
DELAYED_HGO_PER_GRAM = 0.015  # 0.02->0.015 g/hr/gram over threshold, keeps night dip below day.
DELAYED_HGO_MAX_BUMP = 3.5  # [DIURNAL] 5.0->3.5 Cap on HGO bump magnitude (g/hr)
DELAYED_HGO_DELAY_HOURS_MIN = 3.5  # Earliest onset after meal
DELAYED_HGO_DELAY_HOURS_MAX = 5.5  # Latest onset
DELAYED_HGO_DURATION_HOURS_MIN = 2.7  # [HIVAR 2x] 3.0→2.7 — rebound-duration span
DELAYED_HGO_DURATION_HOURS_MAX = 7.0  # [HIVAR 2x] 6.0→7.0 — rebound-duration span
DELAYED_HGO_RAMP_HOURS = 1.0  # Trapezoidal ramp up/down for the rebound envelope

# Per-step absorption noise (gut/depot variance); multiplicative, only matters when curve nonzero.
CARB_ABSORPTION_NOISE_SIGMA = 0.0  # [MINNOISE] step-to-step carb-absorption noise minimized to zero
INSULIN_ABSORPTION_NOISE_SIGMA = 0.0  # [MINNOISE] removed; basal/bolus curves were sawtoothing.

# BG computation
BG_SCALE_FACTOR = 3.5  # 1.5->3.5: Sg now supplies in-band restore; 1.5 over-damped corrections too.
BG_CLAMP_MIN = 10.0  # CGM reading floor, keeps the log risk transform defined; true BG has none.
BG_CLAMP_MAX = 400.0  # [HIVAR] 500→400 — match real CGM device ceiling (Ohio/AZT1D max = 400).
BG_INITIAL_MEAN = 120.0
BG_INITIAL_SIGMA = 60.0  # [HIVAR] 30→60 — wider warmup start (washes out over the run).
BG_INITIAL_FLOOR = 40.0  # Clipping initial draw at CLAMP_MIN starts patients in severe hypo (~4%).

# Soft ceiling: past it a step closes at most SOFT_APPROACH_FRACTION of the headroom to the clamp.
BG_SOFT_CEILING = 385.0        # [HIVAR] 400→385 — keep a soft runway below the new 400 hard ceiling
SOFT_APPROACH_FRACTION = 0.15  # [DAMP] 0.3→0.15 — stronger delta-damping as BG nears the bounds.

# BG regulatory computation
RENAL_THRESHOLD = 180.0  # Kidneys start excreting glucose above this
RENAL_CLEARANCE_RATE = 0.0025  # per step, times mg/dL above RENAL_THRESHOLD; UVA/Padova ke1
COUNTER_REGULATORY_THRESHOLD = 70.0  # Body releases glucagon below this
# Weak: per-unit insulin response stays within 15% between starting BG 90-120 and 160-220.
COUNTER_REGULATORY_RATE = 0.2  # mg/dL per step at BG 0, linear up to the threshold
SEVERE_HYPO_THRESHOLD = 55.0  # Below this, glucagon dump kicks in
SEVERE_HYPO_GLUCAGON_RATE = 0.2  # Extra mg/dL per step at severity=1.0

# Glucose effectiveness (Bergman Sg): always-on insulin-independent pull to a stochastic OU target.

# bg_delta += glucose_effectiveness*(E-bg); E reverts (GE_EQ_TAU_HOURS) to anchor+dawn lift, floors.

# Sg is the in-band (70-180) restore the renal/counter-regulatory guardrails don't cover (acf8h~0).

# Sg is weak by design: high-passes slow inputs incl. insulin; E's timescale drives decorrelation.

# Per-patient Sg lognormal around GE_RATE (~2-3x range); FLOOR keeps the pull up in a severe low.
GE_RATE = 0.0  # insulin-independent pull; zero, so below RENAL_THRESHOLD only insulin lowers BG
GE_EQ_ANCHOR_MEAN = 138.0  # Mean anchor; co-tuned w/ SIGMA/FLOOR to land pooled mean ~162 (Ohio).
GE_EQ_ANCHOR_SIGMA = 15.0  # 12->15: widens mean-BG spread; alone saturates ~14 (floor compresses).
GE_ANCHOR_IR_COUPLING = 30.0  # 12->30 (IR sigma 0.16->0.26): resistant patients run higher.
GE_EQ_SIGMA = 30.0  # mg/dL stationary sd of the equilibrium; scaled per patient by ge_sigma_mult
# Per-patient multiplier on GE_EQ_SIGMA (lognormal); patients shared one within-variance before.
GE_SIGMA_REL_SIGMA = 0.16  # 0.25->0.16: per-patient sigma multiplier; tamed to bound the hypo tail.
GE_SIGMA_MULT_CLIP = (0.68, 1.38)  # clip the multiplier's tails
GE_EQ_TAU_HOURS = 0.75  # equilibrium mean-reversion time; short wander raises excursion count
GE_EQ_DAY_BOOST = 10.0  # Legacy flat daytime lift; superseded by the dawn-phenomenon profile below.
# Dawn rhythm replaces the flat day boost with a per-patient daily profile tied to the dawn trait.

# Deterministic in time-of-day and trait, so a downstream model can learn it (OU noise can't).

# Co-tuned down against GE_EQ_SIGMA so total spread stays on Ohio; periodic variance share rises.
GE_EQ_DAWN_AMPLITUDE_MEAN = 35.0  # Mean daily-rhythm amplitude (mg/dL); scales with the dawn trait.
GE_DAWN_PEAK_HOUR = 5.0  # clock hour the equilibrium's daily profile peaks
GE_DAWN_WIDTH_HOURS = 5.5  # Broad Gaussian width so daytime stays elevated; only small hours dip.
GE_DAY_START_HOUR = 7.0    # setpoint ramps up around wake
GE_DAY_END_HOUR = 22.0     # ramps back down in the late evening
GE_DAY_RAMP_HOURS = 3.0    # smootherstep ramp width for the day/night setpoint transitions
GE_REL_SIGMA = 0.30  # per-patient lognormal spread of Sg around GE_RATE (~2x inter-individual).
GE_RATE_MIN = 0.0          # floor of the per-patient draw
GE_RATE_MAX = 0.150  # Raised so the strong per-patient Sg (lognormal around GE_RATE) isn't clipped.
GE_EQ_FLOOR = 64.0  # 75->60->64: kept above SEVERE_HYPO_THRESHOLD=55 so the pull stays up in a low.

# CGM lag: first-order diffusion dIG/dt=(BG-IG)/tau (Rebrin/Steil), applied before sensor noise.

# Per-patient lag: firmware compensation varies by sensor gen; a fixed 15min overfit one device.

# Drawing lag per patient makes the model lag-robust rather than tuned to one sensor generation.
CGM_LAG_MEAN_MINUTES = 8.0  # Population mean; below physiological 15min since firmware compensates.
CGM_LAG_SIGMA_MINUTES = 4.0  # between-patient spread over sensor generation and physiology
CGM_LAG_CLIP = (0.0, 20.0)  # 0=fully compensated (reports plasma), 20=uncompensated and slow
CGM_NOISE_FRACTION = 0.060  # 0.120->0.060: reverted to value calibrated for ~5.8 mg/dL d5min-std.
# ~9 mg/dL drift at BG=150; AR(1)-correlated wobble like real CGM, not white-noise per-step spikes.

# AR(1) smooths noise: rho=0.85 metabolic (~22min half-life), rho=0.92 sensor (~42min, Dexcom-like).
NOISE_AR1_RHO_METABOLIC = 0.85
NOISE_AR1_RHO_SENSOR = 0.70  # short-lag sensor noise; sets the 15-min sample entropy
# sqrt(1-rho^2): per-step innovation preserving stationary variance in x_t=rho*x_t-1+eps*scale.
NOISE_AR1_INNOV_METABOLIC = float(np.sqrt(1.0 - NOISE_AR1_RHO_METABOLIC ** 2))
NOISE_AR1_INNOV_SENSOR = float(np.sqrt(1.0 - NOISE_AR1_RHO_SENSOR ** 2))

# Rare events
RARE_EVENT_PROBABILITY = 0.01  # chaotic-day rate; its 3x meal jitter reaches the night window
RARE_EVENT_SKILL_REDUCTION = 0.3  # Even skilled people have bad days sometimes

# Post-hypo basal stand-down prevents cascading corrections; patients suspend basal, not carbs.
HYPO_CORRECTION_REFRACTORY_MIN = 20.0  # Min minutes between hypo corrections (moderate hypo 55-70).
SEVERE_HYPO_REFRACTORY_MIN = 10.0  # Shorter gap for severe hypo (<55); first rescue's carbs act.
# Without this gap, rage-eating stacked 3-5 doses (60+g), sawtoothing hypo vs overcorrection peaks.
POST_HYPO_BASAL_SUSPEND_DURATION_HOURS = 6.0  # sin^2 caps 10-min change ~3-6%; was 2h (56% drops).
POST_HYPO_BASAL_SUSPEND_FACTOR = 0.65  # Peak basal mult; shallower than legacy, same integral.
POST_HYPO_BASAL_SUSPEND_RAMP_MIN = 20.0  # Unused legacy ramp param; superseded by sin^2 envelope.

# Rage behavior
RAGE_EAT_BG_THRESHOLD = 50.0       # Below this, patient may rage eat
RAGE_EAT_CARB_MIN = 12.0           # Minimum rage eat carbs
RAGE_EAT_CARB_MAX = 30.0           # Maximum rage eat carbs
RAGE_EAT_PROBABILITY_BASE = 0.10   # Base chance of rage eating when below threshold
# Rule-of-15 recheck: without it, a deep low cascades (16 rescues, 253g in the worst episode).

# COB awareness baseline is lower than insulin's IOB 0.7: hypo arithmetic is harder mid-symptom.
COB_AWARENESS_BASE = 0.40          # fraction of carbs-on-board even a careless patient accounts for
COB_AWARENESS_SKILL = 0.50  # additional fraction scaled by dosing_competence (max 0.90 total)

# WEEKDAY / WEEKEND PARAMETERS

SIMULATION_START_DAY_OF_WEEK = 0       # Starting day of week (0=Monday, 6=Sunday)
WEEKEND_WAKE_DELAY_HOURS_MIN = 0.0     # Min extra hours slept in on weekends/holidays
WEEKEND_WAKE_DELAY_HOURS_MAX = 0.5     # Max extra hours slept in on weekends/holidays
WEEKEND_MEAL_JITTER_MULTIPLIER = 1.5   # Meal timing variability multiplier on weekends
WEEKEND_CARB_INCREASE_FRACTION = 0.15  # Fraction by which carb amounts can increase on weekends
WEEKEND_EXERCISE_PROB_MULTIPLIER = 0.8 # Exercise probability multiplier on weekends

# Public holidays (non-weekend working days treated as weekend for behavior)
PUBLIC_HOLIDAYS_PER_YEAR_MIN = 10      # Minimum number of public holidays per year
PUBLIC_HOLIDAYS_PER_YEAR_MAX = 20      # Maximum number of public holidays per year

# EXERCISE: DELAYED INSULIN SENSITIVITY EFFECT

EXERCISE_IS_REDUCTION = 0.10           # IS reduction fraction post-exercise (10% more sensitive)
EXERCISE_IS_DURATION_HOURS = 6.0  # Long enough for evening-hypo; short of a half-day ACF tail.
EXERCISE_IS_RAMP_HOURS = 1.0           # Trapezoidal ramp up/down for the IS boost envelope

# TREND-BASED ANTICIPATORY CORRECTIONS

TREND_CORRECTION_WINDOW_STEPS = 6      # BG history window for trend (6 steps = 30 min)
TREND_LOW_RATE_THRESHOLD = -5.0        # mg/dL/step falling trend to trigger preemptive carb
TREND_LOW_BG_MAX = 110.0  # Attentive patients eat preemptively while falling through 90-110.

# ALCOHOL MODELING

ALCOHOL_PROBABILITY_WEEKDAY = 0.05     # Per-day drinking probability on weekdays
ALCOHOL_PROBABILITY_WEEKEND = 0.20     # Per-day drinking probability on weekends
ALCOHOL_PROBABILITY_HOLIDAY = 0.30     # Per-day drinking probability on holidays
ALCOHOL_HGO_REDUCTION_MIN = 0.30       # Minimum HGO suppression fraction from alcohol
ALCOHOL_HGO_REDUCTION_MAX = 0.70       # Maximum HGO suppression fraction from alcohol
ALCOHOL_ONSET_DELAY_HOURS_MIN = 1.0    # Hours after drinking before HGO suppression starts
ALCOHOL_ONSET_DELAY_HOURS_MAX = 2.0    # Hours from drinking to end of onset window
ALCOHOL_DURATION_HOURS_MIN = 4.0       # Minimum hours of HGO suppression
ALCOHOL_DURATION_HOURS_MAX = 8.0       # Maximum hours of HGO suppression
ALCOHOL_HGO_RAMP_HOURS = 1.0           # Trapezoidal ramp up/down for HGO suppression envelope

# STRESS AND HORMONAL EFFECTS

STRESS_PROBABILITY_BASE = 0.18         # Per-day base probability of a stress event
STRESS_LIFESTYLE_WEIGHT = 0.16         # How much lifestyle_consistency reduces stress prob
STRESS_IS_FACTOR_MIN = 1.2             # Minimum IS multiplier during stress (more resistant)
STRESS_IS_FACTOR_MAX = 1.5  # Capped low: larger spikes trigger post-stress hypos as they subside.
STRESS_DURATION_HOURS_MIN = 2.0        # Minimum duration of elevated IS from stress (hours)
STRESS_DURATION_HOURS_MAX = 6.0        # Maximum duration of elevated IS from stress (hours)
STRESS_IS_RAMP_HOURS = 0.5             # Trapezoidal ramp up/down for stress envelope

# ANOMALOUS EVENTS

ANOMALOUS_EVENT_PROBABILITY = 0.01     # Per-day probability of an anomalous curve modification
ANOMALOUS_THETA_MULT_MIN = 1.5         # Min theta multiplier (slower absorption)
ANOMALOUS_THETA_MULT_MAX = 3.0         # Max theta multiplier (much slower absorption)
ANOMALOUS_K_MULT_MIN = 0.3             # Min k multiplier (flatter curve)
ANOMALOUS_K_MULT_MAX = 2.0             # Max k multiplier (sharper peak)

# DATA STRUCTURES

class CarbType(Enum):
    FAST = "fast"
    SLOW = "slow"


@dataclass
class PatientProfile:
    """A virtual patient's skill profile and derived parameters."""
    # Raw skills (0-1 after sigmoid)
    dietary_discipline: float = 0.5
    attentiveness: float = 0.5
    dosing_competence: float = 0.5
    lifestyle_consistency: float = 0.5

    # Patient's own treat-low threshold (HYPO_THRESHOLD_MEDIAN); gates the rescue AND every bolus.
    hypo_threshold: float = HYPO_THRESHOLD_MEDIAN

    # This patient's CGM interstitial lag (see CGM_LAG_MEAN_MINUTES); a sensor/physiology property.
    cgm_lag_minutes: float = CGM_LAG_MEAN_MINUTES

    # Derived physiological parameters
    body_weight_kg: float = BODY_WEIGHT_MEAN_KG
    insulin_resistance_factor: float = 1.0  # >1 = resistant, <1 = sensitive
    is_base: float = 1.0
    icr: float = 10.0
    correction_factor: float = 40.0
    glucose_effectiveness: float = GE_RATE  # per-patient Bergman Sg (per-step reversion)
    ge_anchor: float = GE_EQ_ANCHOR_MEAN    # per-patient OU equilibrium anchor (+ diurnal lift)
    ge_sigma_mult: float = 1.0              # per-patient GE_EQ_SIGMA multiplier (within-patient sd)
    basal_dose: float = 20.0

    # One bolus analogue + one basal analogue per patient (BOLUS_VARIANTS/BASAL_VARIANTS).
    bolus_type: str = "aspart"
    basal_type: str = "glargine"
    bolus_gamma_k: float = BOLUS_GAMMA_K
    bolus_gamma_theta: float = BOLUS_GAMMA_THETA
    bolus_dia_base_hours: float = BOLUS_DIA_BASE_HOURS
    basal_ka: float = BASAL_KA_PER_HOUR
    basal_ke: float = BASAL_KE_PER_HOUR
    basal_duration_hours: float = BASAL_DURATION_HOURS  # PK action duration (h)
    basal_tail_clip_hours: float = BASAL_TAIL_CLIP_HOURS
    basal_dose_interval_hours: float = BASAL_DOSE_INTERVAL_HOURS  # injection cadence (h)

    dawn_hgo_amplitude: float = DAWN_HGO_AMPLITUDE_MEAN
    night_hgo_dip_amplitude: float = NIGHT_HGO_DIP_AMPLITUDE_MEAN
    ge_dawn_amplitude: float = GE_EQ_DAWN_AMPLITUDE_MEAN
    exercise_duration_mean_min: float = EXERCISE_DURATION_MEAN_MIN

    # Derived behavioral parameters
    wake_time_hours: float = 8.0
    sleep_duration_hours: float = 7.5
    slow_carb_preference: float = 0.5
    meal_appetite: float = 1.0
    cgm_check_interval_min: float = 60.0
    exercise_probability: float = 0.5
    panic_factor: float = 1.0
    basal_miss_prob: float = 0.01
    meal_jitter_sigma_min: float = 30.0


@dataclass
class ActiveCurve:
    """A time-domain curve (carb absorption, insulin action, etc.)."""
    start_time_idx: int  # Start index in global timeline
    values: np.ndarray  # Curve values at each DT step
    curve_type: str  # 'carb', 'insulin', 'exercise', 'hgo', 'correction_carb'
    label: str = ""  # Human-readable label


@dataclass
class SimulatorState:
    """Current state of the simulation."""
    current_idx: int = 0  # Current time index
    bg: float = 120.0  # Current true BG
    bg_observed: float = 120.0  # CGM reading
    active_curves: list = field(default_factory=list)  # Kept for external access only
    bg_history: list = field(default_factory=list)
    bg_obs_history: list = field(default_factory=list)
    carb_curve_history: list = field(default_factory=list)
    insulin_curve_history: list = field(default_factory=list)
    resistance_history: list = field(default_factory=list)
    exercise_curve_history: list = field(default_factory=list)
    hgo_history: list = field(default_factory=list)
    delta_history: list = field(default_factory=list)
    is_sick: bool = False
    illness_is_factor: float = 1.0
    last_cgm_check_idx: int = 0
    day_number: int = 0
    is_rare_event_day: bool = False
    illness_is_target: float = 1.0
    # Weekday/weekend/holiday tracking
    day_of_week: int = 0               # 0=Monday ... 6=Sunday
    is_holiday: bool = False           # Whether today is a public holiday
    # Time-limited physiological effects
    exercise_is_effects: list = field(default_factory=list)  # (start_idx, end_idx, reduction)
    alcohol_effects: list = field(default_factory=list)      # (start_idx, end_idx, hgo_factor)
    stress_effects: list = field(default_factory=list)       # (start_idx, end_idx, is_factor)
    meal_hgo_effects: list = field(default_factory=list)  # (start_idx, end_idx, magnitude_g_per_hr)
    # Slow physiological state
    glycogen_grams: float = 70.0  # Current hepatic glycogen reserve (g)
    glucotox_bg_ema: float = 120.0  # 3h EMA of true BG, drives glucotoxic IR
    # Hypo correction tracking (see HYPO_CORRECTION_REFRACTORY_MIN).
    last_hypo_correction_idx: int = -9999
    # (start_idx, until_idx) suspend envelopes; effective factor is min across overlaps, never up.
    post_hypo_basal_suspend_windows: list = field(default_factory=list)
    # Next scheduled basal injection idx (-1=uninitialised); advances by BASAL_DOSE_INTERVAL_HOURS.
    next_basal_due_idx: int = -1


# CURVE GENERATION UTILITIES

GAMMA_CURVE_SUBSTEPS = 16  # sub-samples per step when integrating a gamma curve below


def gamma_curve(total_amount: float, k: float, theta: float,
                duration_minutes: float, dt: float = DT_MINUTES) -> np.ndarray:
    """
    Generate a gamma-distributed absorption/action curve.
    Area under curve = total_amount.
    """
    n_steps = int(duration_minutes / dt)
    if n_steps <= 0:
        return np.array([0.0])
    # Integrates the gamma across each step, not right-edge sample: else it stair-steps at start.
    sub = GAMMA_CURVE_SUBSTEPS
    t = (np.arange(n_steps * sub) + 0.5) * (dt / sub)  # sub-step midpoints from 0
    values = (t ** (k - 1) * np.exp(-t / theta)).reshape(n_steps, sub).mean(axis=1)
    # Normalize so the sum of the array equals total_amount (amount per step)
    area = np.sum(values)
    if area > 0:
        values = values * (total_amount / area)
    return values


def basal_curve(total_amount: float, duration_minutes: float,
                ka_per_hour: float = BASAL_KA_PER_HOUR,
                ke_per_hour: float = BASAL_KE_PER_HOUR,
                tail_clip_hours: float = BASAL_TAIL_CLIP_HOURS,
                dt: float = DT_MINUTES,
                ramp_up_hours: Optional[float] = None,
                ramp_down_hours: Optional[float] = None) -> np.ndarray:
    """Long-acting basal insulin curve (Bateman one-compartment PK): f(t)=exp(-ke*t)-exp(-ka*t).
    Broad peak at tmax=ln(ka/ke)/(ka-ke) (~6.3h default); tail-clip window zeros the end smoothly.
    ramp_up/down_hours are legacy args, ignored. sum(values) = total_amount.
    """
    del ramp_up_hours, ramp_down_hours  # accepted for legacy callers; not used

    n_steps = int(duration_minutes / dt)
    if n_steps <= 0:
        return np.array([0.0])

    ka = max(ke_per_hour + 1e-3, float(ka_per_hour))
    ke = float(ke_per_hour)

    t_h = np.arange(n_steps) * (dt / 60.0)
    curve = np.exp(-ke * t_h) - np.exp(-ka * t_h)
    np.maximum(curve, 0.0, out=curve)

    tail_steps = int(tail_clip_hours * 60 / dt)
    if 0 < tail_steps < n_steps:
        s = np.linspace(1.0, 0.0, tail_steps)
        window = s * s * s * (s * (s * 6.0 - 15.0) + 10.0)
        curve[-tail_steps:] *= window

    area = float(np.sum(curve))
    if area > 0.0:
        curve *= total_amount / area
    return curve


def bolus_pk_for_dose(dose_units: float,
                      gamma_k: float = BOLUS_GAMMA_K,
                      gamma_theta: float = BOLUS_GAMMA_THETA,
                      dia_base_hours: float = BOLUS_DIA_BASE_HOURS) -> tuple:
    """Return (k, theta, duration_minutes) for a bolus of the given dose.
    DIA scales with dose (larger depots dissolve slower, peak later); centered on a 5U reference.
    Defaults are the legacy bolus but are overridden per patient by the analogue (BOLUS_VARIANTS).
    """
    dose = max(0.5, dose_units)
    sqrt_excess = float(np.sqrt(dose) - np.sqrt(5.0))
    duration_h = float(np.clip(
        dia_base_hours + BOLUS_DIA_DOSE_SCALE * sqrt_excess,
        BOLUS_DIA_MIN_HOURS, BOLUS_DIA_MAX_HOURS,
    ))
    theta = gamma_theta * (1.0 + BOLUS_THETA_DOSE_SLOPE * sqrt_excess)
    return gamma_k, theta, duration_h * 60.0


def ge_day_weight(hour_of_day: float) -> float:
    """Smootherstep 0..1 waking-hours weight for the diurnal lift of the
    glucose-effectiveness equilibrium anchor (1 through the day, 0 overnight).
    The anchor is raised by GE_EQ_DAY_BOOST * this weight to keep the daytime
    plateau, without imposing a fixed daytime target the OU wander can't leave.
    """
    def _sstep(x: float, a: float, b: float) -> float:
        if b <= a:
            return 1.0 if x >= b else 0.0
        t = min(1.0, max(0.0, (x - a) / (b - a)))
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)

    ramp_up = _sstep(hour_of_day, GE_DAY_START_HOUR, GE_DAY_START_HOUR + GE_DAY_RAMP_HOURS)
    ramp_down = 1.0 - _sstep(hour_of_day, GE_DAY_END_HOUR - GE_DAY_RAMP_HOURS, GE_DAY_END_HOUR)
    return ramp_up * ramp_down


def ge_diurnal_profile(hour_of_day: float) -> float:
    """Mean-zero daily (dawn-phenomenon) profile for the OU equilibrium: wrapped Gaussian peaking
    at GE_DAWN_PEAK_HOUR, mean-subtracted so it does not shift the pooled mean. Deterministic in
    hour-of-day -> shows up as 24h ACF/diurnal amplitude, unlike opaque OU noise. A transient peak
    (not a sustained plateau) survives the slow basal-adjustment feedback, which absorbs the latter.
    """
    d = ((hour_of_day - GE_DAWN_PEAK_HOUR + 12.0) % 24.0) - 12.0
    raw = float(np.exp(-0.5 * (d / GE_DAWN_WIDTH_HOURS) ** 2))
    daily_mean = GE_DAWN_WIDTH_HOURS * 2.5066282746310002 / 24.0
    return raw - daily_mean


def envelope_intensity(time_idx: int, start_idx: int, end_idx: int,
                       ramp_up_steps: int, ramp_down_steps: int) -> float:
    """Trapezoidal envelope for time-bounded effects: 0 outside [start_idx, end_idx), ramps
    0->1 over ramp_up_steps, plateaus at 1, ramps back over ramp_down_steps. Softens on/off
    transitions (exercise/stress IS, alcohol HGO) so BG curves show no step discontinuities.
    """
    if time_idx < start_idx or time_idx >= end_idx:
        return 0.0
    progress = time_idx - start_idx
    remaining = end_idx - time_idx
    intensity = 1.0
    if ramp_up_steps > 0 and progress < ramp_up_steps:
        intensity = min(intensity, progress / ramp_up_steps)
    if ramp_down_steps > 0 and remaining < ramp_down_steps:
        intensity = min(intensity, remaining / ramp_down_steps)
    return max(0.0, intensity)


def compute_hgo_rate(insulin_per_step: float) -> float:
    """Hill-function HGO rate (g/hr) given current plasma insulin per step: HGO = SUPPRESSED +
    (UNSUPPRESSED-SUPPRESSED)/(1+insulin/HALF_MAX). Tuned so typical basal (~0.086 U/step, ICR=8)
    yields exactly HGO_BASE (8.25 g/hr), preserving the basal-balances-HGO invariant.
    """
    span = HGO_UNSUPPRESSED_GRAMS_PER_HOUR - HGO_SUPPRESSED_FLOOR_GRAMS_PER_HOUR
    suppression = 1.0 / (1.0 + max(0.0, insulin_per_step) / HGO_INSULIN_HALF_MAX)
    return HGO_SUPPRESSED_FLOOR_GRAMS_PER_HOUR + span * suppression


def delivered_gain_per_unit(icr: float, is_base: float, glucose_effectiveness: float,
                            basal_per_step: float, weight_factor: float) -> float:
    """mg/dL true-BG drop CORRECTION_HORIZON_STEPS after 1 U of aspart: peripheral action plus
    Hill HGO suppression, each decayed by glucose effectiveness. Ignores diurnal IS and noise."""
    av = BOLUS_VARIANTS["aspart"]
    k, theta, dur = bolus_pk_for_dose(1.0, av["gamma_k"], av["gamma_theta"], av["dia_base_hours"])
    n = CORRECTION_HORIZON_STEPS
    curve = gamma_curve(1.0, k, theta, dur)[:n]
    action = np.pad(curve, (0, n - len(curve)))
    hgo_at_basal = compute_hgo_rate(basal_per_step)
    smoothed, drop = 0.0, np.empty(n)
    for t in range(n):
        smoothed += HGO_INSULIN_SMOOTHING_ALPHA * (action[t] - smoothed)
        hgo_cut = (hgo_at_basal - compute_hgo_rate(basal_per_step + smoothed)) * weight_factor
        drop[t] = action[t] * icr / is_base + hgo_cut * DT_MINUTES / 60.0
    decay = (1.0 - glucose_effectiveness) ** (n - 1 - np.arange(n))
    return float(BG_SCALE_FACTOR * np.sum(drop * decay))


CORRECTION_FACTOR_MEAN = delivered_gain_per_unit(
    ICR_MEAN, 1.0, GE_RATE, HGO_BASE_GRAMS_PER_HOUR * 24.0 / ICR_MEAN / STEPS_PER_DAY, 1.0)


# PATIENT GENERATOR

def hgo_surplus_grams_per_day(p: PatientProfile, bolus_units_per_day: float,
                              basal_units_per_day: float) -> float:
    """Liver glucose per day the basal dose leaves uncleared, at the patient's mean insulin level.
    Ignores glycogen gating, alcohol and noise."""
    weight_factor = p.body_weight_kg / BODY_WEIGHT_MEAN_KG
    insulin_per_step = (basal_units_per_day + bolus_units_per_day) / STEPS_PER_DAY
    hepatic = compute_hgo_rate(insulin_per_step) * 24.0 * weight_factor
    circadian = np.sqrt(2.0 * np.pi) * (p.dawn_hgo_amplitude * DAWN_HGO_SIGMA_HOURS
                                        - p.night_hgo_dip_amplitude * NIGHT_HGO_DIP_SIGMA_HOURS)
    return hepatic + circadian - basal_units_per_day * p.icr / p.is_base


def generate_patient(rng: np.random.Generator) -> PatientProfile:
    """Sample a patient from the population."""
    # Build covariance matrix
    n_skills = 4

    cov = np.full((n_skills, n_skills), SKILL_CORRELATION * SKILL_VARIANCE)
    np.fill_diagonal(cov, SKILL_VARIANCE)

    # Sample raw skills from multivariate normal
    raw_skills = rng.multivariate_normal(np.zeros(n_skills), cov)
    # Sigmoid to (0, 1)
    skills = 1.0 / (1.0 + np.exp(-raw_skills))

    skills = np.clip(skills, SKILL_MIN, SKILL_MAX)

    s1, s2, s3, s4 = skills

    profile = PatientProfile()
    profile.dietary_discipline = s1
    profile.attentiveness = s2
    profile.dosing_competence = s3
    profile.lifestyle_consistency = s4
    _dev = (s2 + s3) / 2.0 - HYPO_THRESHOLD_SKILL_MID
    _gain = HYPO_THRESHOLD_SKILL_SPAN / (
        (HYPO_THRESHOLD_SKILL_MID - SKILL_MIN) if _dev < 0.0
        else (SKILL_MAX - HYPO_THRESHOLD_SKILL_MID))
    profile.hypo_threshold = HYPO_THRESHOLD_MEDIAN + _gain * _dev
    profile.cgm_lag_minutes = float(np.clip(
        rng.normal(CGM_LAG_MEAN_MINUTES, CGM_LAG_SIGMA_MINUTES), *CGM_LAG_CLIP))

    # Body weight + insulin resistance are two independent axes spanning real-T1D TDD range.
    profile.body_weight_kg = float(np.clip(
        rng.normal(BODY_WEIGHT_MEAN_KG, BODY_WEIGHT_SIGMA_KG),
        BODY_WEIGHT_MIN_KG, BODY_WEIGHT_MAX_KG))
    profile.insulin_resistance_factor = float(np.clip(
        np.exp(rng.normal(0.0, IR_LOGNORMAL_SIGMA)),
        IR_FACTOR_MIN, IR_FACTOR_MAX))

    # is_base/ICR couple to insulin_resistance_factor (physiologically correlated) plus small noise.
    ir = profile.insulin_resistance_factor
    profile.is_base = max(0.3, ir * np.exp(rng.normal(0.0, IR_TO_IS_NOISE_SIGMA)))
    profile.icr = max(3.0, (ICR_MEAN / ir) * np.exp(rng.normal(0.0, IR_TO_ICR_NOISE_SIGMA)))
    # Per-patient Sg, lognormal (~2-3x real spread); floored so no patient is a pure integrator.
    profile.glucose_effectiveness = float(np.clip(
        GE_RATE * np.exp(rng.normal(0.0, GE_REL_SIGMA)),
        GE_RATE_MIN, GE_RATE_MAX))
    # Per-patient equilibrium anchor carries mean-BG heterogeneity; resistant sit a bit higher.
    profile.ge_anchor = float(np.clip(
        rng.normal(GE_EQ_ANCHOR_MEAN + GE_ANCHOR_IR_COUPLING * (ir - 1.0), GE_EQ_ANCHOR_SIGMA),
        110.0, 210.0))
    # Per-patient wander amplitude: without it, GE_EQ_SIGMA made every patient equally variable.
    profile.ge_sigma_mult = float(np.clip(
        np.exp(rng.normal(0.0, GE_SIGMA_REL_SIGMA)),
        GE_SIGMA_MULT_CLIP[0], GE_SIGMA_MULT_CLIP[1]))
    profile.dawn_hgo_amplitude = max(0.0, rng.normal(DAWN_HGO_AMPLITUDE_MEAN, DAWN_HGO_AMPLITUDE_SIGMA))
    profile.night_hgo_dip_amplitude = max(0.0, rng.normal(NIGHT_HGO_DIP_AMPLITUDE_MEAN, NIGHT_HGO_DIP_AMPLITUDE_SIGMA))
    # Dawn equilibrium swing scales with the SAME dawn trait as the HGO surge (0=none, ~2=strong).
    dawn_factor = (min(2.0, profile.dawn_hgo_amplitude / DAWN_HGO_AMPLITUDE_MEAN)
                   if DAWN_HGO_AMPLITUDE_MEAN > 0 else 0.0)
    profile.ge_dawn_amplitude = GE_EQ_DAWN_AMPLITUDE_MEAN * dawn_factor
    profile.exercise_duration_mean_min = float(np.clip(
        rng.normal(EXERCISE_DURATION_MEAN_MIN, EXERCISE_DURATION_MEAN_SIGMA_MIN),
        *EXERCISE_DURATION_MEAN_MIN_CLAMP))

    # Ideal basal = HGO*24*IS/ICR (weight/75-scaled); competent (high s3) patients stay close to it.
    weight_factor = profile.body_weight_kg / BODY_WEIGHT_MEAN_KG
    ideal_basal = (HGO_BASE_GRAMS_PER_HOUR * 24.0) * weight_factor * profile.is_base / profile.icr
    # Strong s3 nonlinearity: high-skill patients get near-perfect basal, skip oscillating feedback.
    noise_scale = BASAL_DOSE_SIGMA * (1.5 - s3) ** 2.5
    # Clamp widened [5,40]->[5,80]: heavy IR patients can need 60+ U basal/day (110kg, IR=1.8).
    profile.basal_dose = float(np.clip(rng.normal(ideal_basal, noise_scale), 5.0, 80.0))
    profile.correction_factor = delivered_gain_per_unit(
        profile.icr, profile.is_base, profile.glucose_effectiveness,
        profile.basal_dose / STEPS_PER_DAY, weight_factor)

    # One bolus + basal analogue per patient (variant tables); both dosed once daily at full dose.
    profile.bolus_type = str(rng.choice(list(BOLUS_VARIANTS.keys())))
    bv = BOLUS_VARIANTS[profile.bolus_type]
    profile.bolus_gamma_k = bv["gamma_k"]
    profile.bolus_gamma_theta = bv["gamma_theta"]
    profile.bolus_dia_base_hours = bv["dia_base_hours"]

    profile.basal_type = str(rng.choice(list(BASAL_VARIANTS.keys())))
    av = BASAL_VARIANTS[profile.basal_type]
    profile.basal_ka = av["ka"]
    profile.basal_ke = av["ke"]
    profile.basal_duration_hours = av["action_hours"]
    profile.basal_tail_clip_hours = av["tail_clip_hours"]
    profile.basal_dose_interval_hours = BASAL_DOSE_INTERVAL_HOURS

    # Behavioral parameters derived from skills
    wake_sigma = WAKE_TIME_SIGMA_BASE / (0.3 + 0.7 * s4)
    profile.wake_time_hours = rng.normal(WAKE_TIME_MEAN_HOURS, wake_sigma)
    profile.sleep_duration_hours = rng.normal(SLEEP_DURATION_MEAN_HOURS, SLEEP_DURATION_SIGMA_HOURS)

    profile.slow_carb_preference = SLOW_CARB_PREFERENCE_BASE + SLOW_CARB_PREFERENCE_SKILL_BONUS * s1
    profile.meal_appetite = float(np.clip(
        np.exp(rng.normal(0.0, MEAL_APPETITE_LOG_SIGMA)),
        MEAL_APPETITE_CLIP[0], MEAL_APPETITE_CLIP[1]))
    profile.cgm_check_interval_min = (CGM_CHECK_INTERVAL_ATTENTIVE +
                                       (CGM_CHECK_INTERVAL_INATTENTIVE - CGM_CHECK_INTERVAL_ATTENTIVE) * (1 - s2))
    profile.exercise_probability = EXERCISE_PROBABILITY_BASE + EXERCISE_SKILL_BONUS * s4
    profile.panic_factor = HYPO_PANIC_FACTOR_BASE * (1.2 - s3)
    profile.basal_miss_prob = BASAL_MISS_PROB_BASE * np.exp(BASAL_MISS_SKILL_SCALE * (0.5 - s2))
    profile.meal_jitter_sigma_min = MEAL_TIME_JITTER_BASE_MIN / (0.2 + 0.8 * s4)

    profile.wake_time_hours = np.clip(profile.wake_time_hours, 4.0, 12.0)

    return profile


# MAIN SIMULATOR

class T1DMSimulator:
    """Generates factor curves and BG trace for a virtual T1DM patient.
    Call generate() repeatedly to advance the simulation by DT_MINUTES.
    Curve contributions pre-accumulate into numpy arrays (O(1) per-step reads); IOB is a numpy
    prefix-sum over the future insulin array.
    """

    def __init__(self, seed: int = 42, initial_bg: Optional[float] = None):
        self.rng = np.random.default_rng(seed)
        # Physiology and sensor noise draw here, so behaviour draws never shift them.
        self.noise_rng = np.random.default_rng([seed, 1])
        self.patient = generate_patient(self.rng)
        self.state = SimulatorState()

        # Set initial BG
        if initial_bg is not None:
            self.state.bg = np.clip(initial_bg, BG_CLAMP_MIN, BG_CLAMP_MAX)
        else:
            skill_avg = (self.patient.dietary_discipline + self.patient.dosing_competence) / 2.0
            bg_mean = BG_INITIAL_MEAN + 40.0 * (0.5 - skill_avg)
            self.state.bg = np.clip(
                self.rng.normal(bg_mean, BG_INITIAL_SIGMA),
                BG_INITIAL_FLOOR, BG_CLAMP_MAX
            )

        self.state.bg_observed = self.state.bg
        self.state.glycogen_grams = GLYCOGEN_CAPACITY_GRAMS * GLYCOGEN_INITIAL_FRACTION
        self.state.glucotox_bg_ema = float(self.state.bg)

        # Holiday tracking
        self._holiday_set: set = set()
        self._holidays_generated_years: set = set()
        self._generate_year_holidays(0)
        self._generate_year_holidays(1)

        # Vectorized accumulators (by step); curves scatter-add here for O(1) generate() reads.
        _init_len = STEPS_PER_DAY * 4
        self._carb_totals: np.ndarray = np.zeros(_init_len)
        self._basal_totals: np.ndarray = np.zeros(_init_len)
        self._bolus_totals: np.ndarray = np.zeros(_init_len)
        self._exercise_totals: np.ndarray = np.zeros(_init_len)
        self._rescue_totals: np.ndarray = np.zeros(_init_len)  # correction_carb only, rule-of-15
        # (start_idx, curve_type, total amount, label) per injected curve, in injection order.
        self.injection_log: list = []

        # EMA-smoothed insulin for the HGO Hill function; models plasma lag behind SC absorption.
        self._smoothed_insulin_for_hgo: float = 0.0

        # AR(1): noise = rho*noise_prev + sqrt(1-rho^2)*N(0,sigma); stationary var sigma^2, smooth.
        self._ar_is: float = 0.0
        self._ar_hgo: float = 0.0
        self._ar_carb: float = 0.0
        self._ar_insulin: float = 0.0
        self._ar_cgm: float = 0.0
        # Interstitial glucose state for the CGM lag (see CGM_LAG_MINUTES).
        self._interstitial_bg: float = float(self.state.bg)
        # Ornstein-Uhlenbeck glucose-effectiveness equilibrium (see GE_EQ_*).
        self._ge_equilibrium: float = float(self.patient.ge_anchor)

        # Pre-generate day plan
        self._plan_day()

        # Pending events: list of (time_idx, event_type, event_data)
        self._pending_events: list = []
        self._generate_day_events()

    def reseed(self, seed: int, initial_bg: Optional[float] = None):
        """Reset the simulator with a new seed."""
        self.rng = np.random.default_rng(seed)
        # Physiology and sensor noise draw here, so behaviour draws never shift them.
        self.noise_rng = np.random.default_rng([seed, 1])
        self.patient = generate_patient(self.rng)
        self.state = SimulatorState()

        if initial_bg is not None:
            self.state.bg = np.clip(initial_bg, BG_CLAMP_MIN, BG_CLAMP_MAX)
        else:
            skill_avg = (self.patient.dietary_discipline + self.patient.dosing_competence) / 2.0
            bg_mean = BG_INITIAL_MEAN + 40.0 * (0.5 - skill_avg)
            self.state.bg = np.clip(
                self.rng.normal(bg_mean, BG_INITIAL_SIGMA),
                BG_INITIAL_FLOOR, BG_CLAMP_MAX
            )

        self.state.bg_observed = self.state.bg
        self.state.glycogen_grams = GLYCOGEN_CAPACITY_GRAMS * GLYCOGEN_INITIAL_FRACTION
        self.state.glucotox_bg_ema = float(self.state.bg)

        self._holiday_set = set()
        self._holidays_generated_years = set()
        self._generate_year_holidays(0)
        self._generate_year_holidays(1)

        _init_len = STEPS_PER_DAY * 4
        self._carb_totals = np.zeros(_init_len)
        self._basal_totals = np.zeros(_init_len)
        self._bolus_totals = np.zeros(_init_len)
        self._exercise_totals = np.zeros(_init_len)
        self._rescue_totals = np.zeros(_init_len)  # correction_carb only, rule-of-15 recheck
        self.injection_log = []
        self._smoothed_insulin_for_hgo = 0.0

        # Reset AR(1) noise state (mirrors __init__).
        self._ar_is = 0.0
        self._ar_hgo = 0.0
        self._ar_carb = 0.0
        self._ar_insulin = 0.0
        self._ar_cgm = 0.0
        self._interstitial_bg = float(self.state.bg)
        self._ge_equilibrium = float(self.patient.ge_anchor)

        self._pending_events = []

        # Daily IS drift state must clear so day 1 doesn't blend a prior patient's leftover values.
        for attr in ('_daily_is_drift', '_daily_is_phase_shift',
                     '_prev_daily_is_drift', '_prev_daily_is_phase_shift'):
            if hasattr(self, attr):
                delattr(self, attr)

        self._plan_day()
        self._generate_day_events()

    # Internal helpers

    def _ensure_totals_length(self, required_length: int) -> None:
        """Grow accumulation arrays to cover at least required_length steps."""
        current = len(self._carb_totals)
        if required_length > current:
            extra = max(required_length - current, STEPS_PER_DAY)
            self._carb_totals = np.concatenate([self._carb_totals, np.zeros(extra)])
            self._basal_totals = np.concatenate([self._basal_totals, np.zeros(extra)])
            self._bolus_totals = np.concatenate([self._bolus_totals, np.zeros(extra)])
            self._exercise_totals = np.concatenate([self._exercise_totals, np.zeros(extra)])
            self._rescue_totals = np.concatenate([self._rescue_totals, np.zeros(extra)])

    def _add_to_totals(self, curve: np.ndarray, start_idx: int, curve_type: str) -> None:
        """Scatter-add a curve into the appropriate accumulation array.

        After this call, self._carb_totals[start_idx + i] (etc.) contains the
        summed contribution from all curves active at that step.
        """
        n = len(curve)
        end = start_idx + n
        self._ensure_totals_length(end)
        if curve_type in ('carb', 'correction_carb'):
            self._carb_totals[start_idx:end] += curve
            if curve_type == 'correction_carb':
                # Tracked separately: rescue branch sees its own COB, not meal carbs (bolus-linked).
                self._rescue_totals[start_idx:end] += curve
        elif curve_type == 'basal':
            self._basal_totals[start_idx:end] += curve
        elif curve_type in ('bolus', 'insulin'):
            self._bolus_totals[start_idx:end] += curve
        elif curve_type == 'exercise':
            self._exercise_totals[start_idx:end] += curve

    def inject_curve(self, values: np.ndarray, start_idx: int,
                     curve_type: str, label: str = '') -> None:
        """Inject a curve directly into the simulation, instead of appending to state.active_curves
        (e.g. from tests or custom scripts). Updates both the accumulation arrays and active_curves.
        """
        self.state.active_curves.append(ActiveCurve(
            start_time_idx=start_idx,
            values=values,
            curve_type=curve_type,
            label=label
        ))
        self._add_to_totals(values, start_idx, curve_type)
        self.injection_log.append((start_idx, curve_type, float(np.sum(values)), label))

    def _generate_year_holidays(self, year: int) -> None:
        """Generate and store public holidays for the given simulation year.

        Holidays are stratified across the year and never fall on weekends.
        """
        if year in self._holidays_generated_years:
            return
        self._holidays_generated_years.add(year)
        year_start = year * 365
        n_holidays = int(self.rng.integers(PUBLIC_HOLIDAYS_PER_YEAR_MIN,
                                            PUBLIC_HOLIDAYS_PER_YEAR_MAX + 1))
        # Stratified: divide year into n_holidays segments, pick one weekday from each
        segment_size = 365.0 / n_holidays
        for i in range(n_holidays):
            seg_start = int(i * segment_size)
            seg_end = max(seg_start + 1, int((i + 1) * segment_size))
            for _ in range(30):  # max attempts to find a weekday in this segment
                day_of_year = int(self.rng.integers(seg_start, seg_end))
                abs_day = year_start + day_of_year
                dow = (SIMULATION_START_DAY_OF_WEEK + abs_day) % 7
                if dow < 5:  # Not Saturday (5) or Sunday (6)
                    self._holiday_set.add(abs_day)
                    break

    def _plan_day(self):
        """Plan a day's schedule."""
        day = self.state.day_number

        # Ensure holidays exist for this and next year
        current_year = day // 365
        for yr in range(current_year, current_year + 2):
            if yr not in self._holidays_generated_years:
                self._generate_year_holidays(yr)

        # Day-of-week and holiday status
        self.state.day_of_week = (SIMULATION_START_DAY_OF_WEEK + day) % 7
        self.state.is_holiday = day in self._holiday_set

        # Check for rare event day
        rare_prob = RARE_EVENT_PROBABILITY
        self.state.is_rare_event_day = self.rng.random() < rare_prob

        # Check for illness onset/continuation
        if not self.state.is_sick:
            sick_prob = ILLNESS_PROBABILITY_BASE * (1 - ILLNESS_HEALTH_WEIGHT * self.patient.lifestyle_consistency)
            if self.rng.random() < sick_prob:
                self.state.is_sick = True
                self.state.illness_is_target = self.rng.uniform(ILLNESS_IS_FACTOR_MIN, ILLNESS_IS_FACTOR_MAX)
        else:
            if self.rng.random() < ILLNESS_RECOVERY_PROB:
                self.state.is_sick = False
                self.state.illness_is_target = 1.0

        # Gradually ramp illness IS factor toward target
        diff = self.state.illness_is_target - self.state.illness_is_factor
        self.state.illness_is_factor += diff * ILLNESS_IS_RAMP_RATE

        # Daily IS drift blends across midnight; scales w/ (1.5-s4) so chaotic patients swing more.
        self._prev_daily_is_drift = getattr(self, '_daily_is_drift', 0.0)
        self._prev_daily_is_phase_shift = getattr(self, '_daily_is_phase_shift', 0.0)
        drift_sigma = IS_DAILY_DRIFT_SIGMA * (1.5 - self.patient.lifestyle_consistency)
        self._daily_is_drift = self.rng.normal(0, drift_sigma)
        self._daily_is_phase_shift = self.rng.normal(0, IS_DAWN_PHASE_DAILY_SIGMA)

    def _generate_day_events(self):
        """Generate all events for the current day."""
        day_start_idx = self.state.day_number * (24 * 60 // DT_MINUTES)
        p = self.patient
        s = self.state

        is_weekend = s.day_of_week >= 5   # Saturday or Sunday
        is_special_day = is_weekend or s.is_holiday

        # Determine effective skills for today
        if s.is_rare_event_day:
            # On rare days, all skills are degraded
            skill_penalty = RARE_EVENT_SKILL_REDUCTION + self.rng.random() * 0.3
            eff_s1 = max(0.05, p.dietary_discipline - skill_penalty)
            eff_s3 = max(0.05, p.dosing_competence - skill_penalty)
            eff_s4 = max(0.05, p.lifestyle_consistency - skill_penalty)
        else:
            eff_s1 = p.dietary_discipline
            eff_s3 = p.dosing_competence
            eff_s4 = p.lifestyle_consistency

        # Wake time for today — weekends/holidays shift it later
        wake_sigma = WAKE_TIME_SIGMA_BASE / (0.3 + 0.7 * eff_s4)
        if s.is_rare_event_day:
            wake_sigma *= 3.0
        today_wake = float(np.clip(self.rng.normal(WAKE_TIME_MEAN_HOURS, wake_sigma), 4.0, 14.0))
        if is_special_day:
            delay = self.rng.uniform(WEEKEND_WAKE_DELAY_HOURS_MIN, WEEKEND_WAKE_DELAY_HOURS_MAX)
            today_wake = min(14.0, today_wake + delay)

        wake_idx = day_start_idx + int(today_wake * 60 / DT_MINUTES)
        sleep_hours = float(np.clip(
            self.rng.normal(SLEEP_DURATION_MEAN_HOURS, SLEEP_DURATION_SIGMA_HOURS),
            4.0, 12.0,
        ))
        # Bedtime = wake + (24-sleep_hours); floor on awake time avoids instant-sleep-after-wake.
        awake_hours = max(8.0, 24.0 - sleep_hours)
        sleep_idx = day_start_idx + int((today_wake + awake_hours) * 60 / DT_MINUTES)

        # Store wake/sleep for the day
        self._today_wake_idx = wake_idx
        self._today_sleep_idx = sleep_idx

        # --- Anomalous event flag for the day ---
        anomalous_today = self.rng.random() < ANOMALOUS_EVENT_PROBABILITY
        anomalous_applied = False  # Only apply to first eligible event

        # Analogues dose daily at full 24h basal_dose (factor=1.0); action_hours sets PK length.

        # degludec (42h) overlaps ~1.75 doses (flat); glargine (26h) ~1 dose (mild end-of-day wane).
        day_end_idx = day_start_idx + STEPS_PER_DAY
        basal_interval_steps = max(1, int(p.basal_dose_interval_hours * 60 / DT_MINUTES))
        per_dose_factor = p.basal_dose_interval_hours / 24.0

        if s.next_basal_due_idx < 0:
            # First-ever basal anchors at wake_idx (un-jittered; jitter adds +-30min per loop).
            s.next_basal_due_idx = wake_idx

        while s.next_basal_due_idx < day_end_idx:
            jitter_steps = int(self.rng.normal(0, 30) / DT_MINUTES)
            dose_idx = max(self.state.current_idx, s.next_basal_due_idx + jitter_steps)

            if self.rng.random() > p.basal_miss_prob:
                # Administer basal — multiplied by injection-site quality
                dose_noise = 1.0 + self.rng.normal(0, BASAL_DOSE_COMPETENCE_NOISE * (1.2 - eff_s3))
                # Contracts site-quality toward 1.0: basal sites absorb more consistently.
                raw_site_q = self._site_quality(eff_s4)
                site_q = 1.0 + (raw_site_q - 1.0) * BASAL_SITE_QUALITY_DAMPING
                actual_dose = max(0.5, p.basal_dose * per_dose_factor * dose_noise * site_q)
                duration = p.basal_duration_hours * 60
                curve = basal_curve(float(actual_dose), duration,
                                    ka_per_hour=p.basal_ka, ke_per_hour=p.basal_ke,
                                    tail_clip_hours=p.basal_tail_clip_hours)
                self._pending_events.append((dose_idx, 'basal', {
                    'curve': curve,
                    'label': (f'Basal {actual_dose:.1f}U '
                              f'{p.basal_type} ({p.basal_duration_hours:.0f}h)'),
                }))

            # Advances regardless of miss: the patient still intended to inject at this daily slot.
            s.next_basal_due_idx = dose_idx + basal_interval_steps

        # --- Meals ---
        if s.is_rare_event_day:
            if self.rng.random() < 0.3:
                n_meals = max(0, self.rng.poisson(1))
            else:
                n_meals = self.rng.poisson(MEALS_BASE + 2)
        else:
            extra_lambda = MEALS_EXTRA_LAMBDA * (1 - eff_s1)
            if is_special_day:
                extra_lambda *= 1.3
            extra = self.rng.poisson(extra_lambda)
            n_meals = MEALS_BASE + extra

        day_glucose = 0.0
        for i in range(n_meals):
            if i < len(MEAL_TIME_OFFSETS_HOURS):
                offset = MEAL_TIME_OFFSETS_HOURS[i]
                carb_mean = MEAL_CARB_MEANS[i]
            else:
                offset = self.rng.uniform(1, 14)
                carb_mean = SNACK_CARB_MEAN

            # Meal timing jitter: poor dietary discipline adds variance
            jitter_sigma = p.meal_jitter_sigma_min * (1.0 + 0.5 * (1.0 - eff_s1))
            if s.is_rare_event_day:
                jitter_sigma *= 3.0
            if is_special_day:
                jitter_sigma *= WEEKEND_MEAL_JITTER_MULTIPLIER
            jitter = self.rng.normal(0, jitter_sigma)
            meal_time = today_wake + offset + jitter / 60.0
            meal_idx = max(self.state.current_idx, day_start_idx + int(meal_time * 60 / DT_MINUTES))

            # Carb amount
            discipline_factor = 1.0 - MEAL_CARB_DISCIPLINE_SCALE * eff_s1
            weekend_factor = 1.0
            if is_special_day:
                weekend_factor = 1.0 + self.rng.uniform(0, WEEKEND_CARB_INCREASE_FRACTION)
            discipline_carb_sigma = (MEAL_CARB_SIGMA * (1.0 + 0.5 * (1.0 - eff_s1))
                                     * p.meal_appetite)
            carb_amount = max(0.0, self.rng.normal(
                carb_mean * discipline_factor * weekend_factor
                * p.meal_appetite * MEAL_CARB_SCALE, discipline_carb_sigma))

            # Meal = 2-5 overlapping gamma curves (fast/med/slow); weights tilt slow with high s1.
            slow_pref = SLOW_CARB_PREFERENCE_BASE + SLOW_CARB_PREFERENCE_SKILL_BONUS * eff_s1
            fast_w = max(0.05, (1.0 - slow_pref) + self.rng.normal(0, 0.1))
            slow_w = max(0.05, slow_pref + self.rng.normal(0, 0.1))
            med_w = max(0.05, MIXED_MEAL_MED_WEIGHT_BASE + self.rng.normal(0, 0.1))
            type_weights = np.array([fast_w, med_w, slow_w])
            type_weights = type_weights / type_weights.sum()

            n_extra = int(self.rng.poisson(MIXED_MEAL_EXTRA_COMPONENTS_LAMBDA))
            n_components = min(MIXED_MEAL_MAX_COMPONENTS,
                               MIXED_MEAL_MIN_COMPONENTS + n_extra)
            fractions = self.rng.dirichlet(np.full(n_components, MIXED_MEAL_DIRICHLET_ALPHA))
            component_types = self.rng.choice(['fast', 'med', 'slow'],
                                               size=n_components, p=type_weights)

            # Apply anomalous event shape modification to one component this day
            def _maybe_anomalous(k: float, theta: float) -> tuple:
                nonlocal anomalous_applied
                if anomalous_today and not anomalous_applied:
                    anomalous_applied = True
                    k *= float(self.rng.uniform(ANOMALOUS_K_MULT_MIN, ANOMALOUS_K_MULT_MAX))
                    theta *= float(self.rng.uniform(ANOMALOUS_THETA_MULT_MIN, ANOMALOUS_THETA_MULT_MAX))
                return k, theta

            for ctype, frac in zip(component_types, fractions):
                component_carbs = float(carb_amount * frac)
                if ctype == 'fast':
                    k = float(self.rng.uniform(*MIXED_MEAL_FAST_K_RANGE))
                    theta = float(self.rng.uniform(*MIXED_MEAL_FAST_THETA_RANGE))
                elif ctype == 'med':
                    k = float(self.rng.uniform(*MIXED_MEAL_MED_K_RANGE))
                    theta = float(self.rng.uniform(*MIXED_MEAL_MED_THETA_RANGE))
                else:
                    k = float(self.rng.uniform(*MIXED_MEAL_SLOW_K_RANGE))
                    theta = float(self.rng.uniform(*MIXED_MEAL_SLOW_THETA_RANGE))
                k *= (1 + self.rng.normal(0, CARB_CURVE_K_NOISE))
                theta *= (1 + self.rng.normal(0, CARB_CURVE_THETA_NOISE))
                k, theta = _maybe_anomalous(k, theta)
                k = max(1.1, k); theta = max(3.0, theta)
                duration = max(k * theta * 4, 60)
                self._pending_events.append((meal_idx, 'carb', {
                    'curve': gamma_curve(component_carbs, k, theta, duration),
                    'label': f'Meal {component_carbs:.0f}g {ctype}',
                    'meal_key': (s.day_number, i),
                    'meal_grams': carb_amount,
                }))

            day_glucose += carb_amount
            # Delayed HGO rebound: large meals bump HGO 3.5-5.5h later; drives post-meal highs.
            if carb_amount > DELAYED_HGO_MEAL_THRESHOLD_GRAMS:
                excess = carb_amount - DELAYED_HGO_MEAL_THRESHOLD_GRAMS
                magnitude = min(DELAYED_HGO_MAX_BUMP, DELAYED_HGO_PER_GRAM * excess)
                delay_h = self.rng.uniform(DELAYED_HGO_DELAY_HOURS_MIN, DELAYED_HGO_DELAY_HOURS_MAX)
                duration_h = self.rng.uniform(DELAYED_HGO_DURATION_HOURS_MIN, DELAYED_HGO_DURATION_HOURS_MAX)
                rebound_start = meal_idx + int(delay_h * 60 / DT_MINUTES)
                rebound_end = rebound_start + int(duration_h * 60 / DT_MINUTES)
                s.meal_hgo_effects.append((rebound_start, rebound_end, magnitude))
                day_glucose += magnitude * max(0.0, duration_h - DELAYED_HGO_RAMP_HOURS)

        # --- Boluses: the day's units cover its planned glucose; each time and dose is random ---
        units = BOLUS_BALANCE_GAIN * self._daily_bolus_units(day_glucose)
        per_median = float(np.exp(-0.5 * BOLUS_DOSE_LOG_SIGMA ** 2))
        night_median = BOLUS_NIGHT_UNIT_SHARE * units / BOLUS_NIGHT_EVENTS_PER_DAY * per_median
        day_median = (1.0 - BOLUS_NIGHT_UNIT_SHARE) * units / BOLUS_DAY_EVENTS_PER_DAY * per_median
        n_night = int(self.rng.poisson(BOLUS_NIGHT_EVENTS_PER_DAY))
        n_day = int(self.rng.poisson(BOLUS_DAY_EVENTS_PER_DAY))
        for at_night in [True] * n_night + [False] * n_day:
            if at_night:
                hour = BOLUS_NIGHT_START_HOUR + self.rng.uniform(0.0, BOLUS_NIGHT_HOURS)
                median = night_median
            else:
                hour = (BOLUS_NIGHT_START_HOUR + BOLUS_NIGHT_HOURS
                        + self.rng.uniform(0.0, 24.0 - BOLUS_NIGHT_HOURS))
                median = day_median
            bolus_idx = max(s.current_idx, day_start_idx + int((hour % 24.0) * 60 / DT_MINUTES))
            dose = median * float(np.exp(self.rng.normal(0.0, BOLUS_DOSE_LOG_SIGMA)))
            # PK shape follows the intended dose; site quality only modulates absorbed amount.
            base_k, base_theta, bolus_duration = bolus_pk_for_dose(
                dose, p.bolus_gamma_k, p.bolus_gamma_theta, p.bolus_dia_base_hours)
            bolus_k = base_k * (1 + self.rng.normal(0, 0.05))
            bolus_theta = base_theta * (1 + self.rng.normal(0, 0.05))
            delivered_dose = dose * self._site_quality(eff_s4)
            bolus_curve = gamma_curve(delivered_dose, max(1.5, bolus_k),
                                      max(5.0, bolus_theta), bolus_duration)
            self._pending_events.append((bolus_idx, 'bolus', {
                'curve': bolus_curve, 'label': f'Bolus {delivered_dose:.1f}U'
            }))

        # --- Exercise ---
        ex_prob = EXERCISE_PROBABILITY_BASE + EXERCISE_SKILL_BONUS * eff_s4
        if s.is_rare_event_day:
            ex_prob *= 0.3
        if is_special_day:
            ex_prob *= WEEKEND_EXERCISE_PROB_MULTIPLIER

        if self.rng.random() < ex_prob:
            ex_offset = self.rng.normal(EXERCISE_TIME_MEAN_OFFSET_HOURS, EXERCISE_TIME_SIGMA_HOURS)
            ex_time = today_wake + ex_offset
            ex_idx = max(self.state.current_idx, day_start_idx + int(ex_time * 60 / DT_MINUTES))
            ex_duration = max(10.0, self.rng.normal(p.exercise_duration_mean_min, EXERCISE_DURATION_SIGMA_MIN))
            ex_magnitude = ex_duration * EXERCISE_CARB_EQUIV_PER_MIN
            ex_curve_duration = ex_duration + 90
            ex_curve = gamma_curve(ex_magnitude, EXERCISE_GAMMA_K, EXERCISE_GAMMA_THETA, ex_curve_duration)
            self._pending_events.append((ex_idx, 'exercise', {
                'curve': ex_curve,
                'label': f'Exercise {ex_duration:.0f}min',
                'duration_min': ex_duration,  # stored for IS effect scheduling
            }))

        # --- Alcohol event (suppresses HGO, causing delayed lows) ---
        if is_special_day:
            alcohol_prob = ALCOHOL_PROBABILITY_HOLIDAY if s.is_holiday else ALCOHOL_PROBABILITY_WEEKEND
        else:
            alcohol_prob = ALCOHOL_PROBABILITY_WEEKDAY
        if s.is_rare_event_day:
            alcohol_prob = max(alcohol_prob, ALCOHOL_PROBABILITY_WEEKEND)
        alcohol_prob *= (1.2 - eff_s4)

        if self.rng.random() < alcohol_prob:
            drink_offset_hours = MEAL_TIME_OFFSETS_HOURS[-1] + self.rng.uniform(0.0, 2.0)
            drink_time = today_wake + drink_offset_hours
            drink_idx = max(self.state.current_idx, day_start_idx + int(drink_time * 60 / DT_MINUTES))

            onset_delay = self.rng.uniform(ALCOHOL_ONSET_DELAY_HOURS_MIN, ALCOHOL_ONSET_DELAY_HOURS_MAX)
            duration = self.rng.uniform(ALCOHOL_DURATION_HOURS_MIN, ALCOHOL_DURATION_HOURS_MAX)
            hgo_reduction = self.rng.uniform(ALCOHOL_HGO_REDUCTION_MIN, ALCOHOL_HGO_REDUCTION_MAX)
            hgo_factor = 1.0 - hgo_reduction

            start_idx = drink_idx + int(onset_delay * 60 / DT_MINUTES)
            end_idx = start_idx + int(duration * 60 / DT_MINUTES)
            s.alcohol_effects.append((start_idx, end_idx, hgo_factor))

        # --- Stress event (transient increase in insulin resistance) ---
        stress_prob = max(0.01, STRESS_PROBABILITY_BASE - STRESS_LIFESTYLE_WEIGHT * eff_s4)
        if self.rng.random() < stress_prob:
            stress_offset = self.rng.uniform(1.0, 10.0)
            stress_time = today_wake + stress_offset
            stress_idx = max(self.state.current_idx, day_start_idx + int(stress_time * 60 / DT_MINUTES))

            is_factor = self.rng.uniform(STRESS_IS_FACTOR_MIN, STRESS_IS_FACTOR_MAX)
            duration_hours = self.rng.uniform(STRESS_DURATION_HOURS_MIN, STRESS_DURATION_HOURS_MAX)
            end_idx = stress_idx + int(duration_hours * 60 / DT_MINUTES)
            s.stress_effects.append((stress_idx, end_idx, is_factor))

        # Sort events by time
        self._pending_events.sort(key=lambda x: x[0])

    def _daily_bolus_units(self, glucose_grams: float) -> float:
        """Units that clear glucose_grams plus the liver surplus left at that insulin level."""
        p = self.patient
        grams_per_unit = p.icr / p.is_base
        units = glucose_grams / grams_per_unit
        for _ in range(4):
            surplus = hgo_surplus_grams_per_day(p, units, p.basal_dose)
            units = max(0.0, (glucose_grams + surplus) / grams_per_unit)
        return units

    def _site_quality(self, s4: float) -> float:
        """Per-dose injection site absorption multiplier, centered on 1.0 (<1 poorly-absorbing scar,
        >1 hyper-absorbing surge). Low lifestyle_consistency (s4) rotates sites poorly, scattering
        dose-to-dose absorption; scaling is super-linear in (1.5-s4) so high-s4 patients converge
        toward near-perfect absorption (needed to keep TBR/TAR low as correction frequency rises).
        """
        sigma = SITE_QUALITY_SIGMA_BASE * (1.5 - s4) ** 1.8
        return float(np.clip(self.rng.normal(1.0, sigma),
                             SITE_QUALITY_MIN, SITE_QUALITY_MAX))

    def _compute_insulin_resistance(self, time_idx: int, active_carb: float = 0.0) -> float:
        """Insulin resistance factor at a time index: diurnal pattern, daily drift (smoothed across
        midnight), illness factor, exercise/stress envelopes, glucotoxic IR, postprandial IR, noise.
        """
        s = self.state

        # Time of day in hours
        hour = (time_idx * DT_MINUTES / 60.0) % 24.0

        # Smooth-step blend of yesterday's drift/phase over IS_DRIFT_TRANSITION_HOURS avoids a step.
        if hour < IS_DRIFT_TRANSITION_HOURS:
            raw = hour / IS_DRIFT_TRANSITION_HOURS
            blend = 0.5 - 0.5 * np.cos(raw * np.pi)
            drift = self._prev_daily_is_drift * (1 - blend) + self._daily_is_drift * blend
            phase_shift = self._prev_daily_is_phase_shift * (1 - blend) + self._daily_is_phase_shift * blend
        else:
            drift = self._daily_is_drift
            phase_shift = self._daily_is_phase_shift

        # Diurnal IR (higher=less glucose cleared); cortisol peaks raise it, deep sleep lowers it.
        morning = IS_MORNING_AMPLITUDE * np.exp(-0.5 * ((hour - IS_MORNING_PEAK_HOUR - phase_shift) / 2.0) ** 2)
        evening = IS_EVENING_AMPLITUDE * np.exp(-0.5 * ((hour - IS_EVENING_PEAK_HOUR) / 2.5) ** 2)
        night_hour = hour if hour < 12 else hour - 24
        night = -IS_NIGHT_DIP_AMPLITUDE * np.exp(-0.5 * ((night_hour - IS_NIGHT_DIP_HOUR) / 2.0) ** 2)
        diurnal = 1.0 + morning + evening + night

        is_val = self.patient.is_base * diurnal * (1.0 + drift)

        # Illness factor always applies; rests at 1.0 healthy, ramps smoothly so IS doesn't step.
        is_val *= s.illness_is_factor

        # Post-exercise IS reduction (sensitivity rises for hours); trapezoid softens on/off edges.
        ex_ramp_steps = int(EXERCISE_IS_RAMP_HOURS * 60 / DT_MINUTES)
        exercise_reduction = 0.0
        active_ex_effects = []
        for (start_idx, end_idx, reduction) in s.exercise_is_effects:
            if time_idx < end_idx:
                active_ex_effects.append((start_idx, end_idx, reduction))
                intensity = envelope_intensity(time_idx, start_idx, end_idx,
                                                ex_ramp_steps, ex_ramp_steps)
                if intensity > 0:
                    exercise_reduction += reduction * intensity
        s.exercise_is_effects = active_ex_effects
        if exercise_reduction > 0:
            is_val *= (1.0 - min(0.30, exercise_reduction))

        # Stress IS effect (transient resistance, cortisol/adrenaline); envelope blends around 1.0.
        stress_ramp_steps = int(STRESS_IS_RAMP_HOURS * 60 / DT_MINUTES)
        stress_factor = 1.0
        active_stress = []
        for (start_idx, end_idx, factor) in s.stress_effects:
            if time_idx < end_idx:
                active_stress.append((start_idx, end_idx, factor))
                intensity = envelope_intensity(time_idx, start_idx, end_idx,
                                                stress_ramp_steps, stress_ramp_steps)
                if intensity > 0:
                    eff_factor = 1.0 + (factor - 1.0) * intensity
                    stress_factor = max(stress_factor, eff_factor)
        s.stress_effects = active_stress
        if stress_factor > 1.0:
            is_val *= stress_factor

        # Glucotoxicity: sustained hyper raises IR via 3h BG EMA above GLUCOTOX_BG_THRESHOLD.
        if s.glucotox_bg_ema > GLUCOTOX_BG_THRESHOLD:
            excess = s.glucotox_bg_ema - GLUCOTOX_BG_THRESHOLD
            span = GLUCOTOX_BG_FOR_MAX - GLUCOTOX_BG_THRESHOLD
            intensity = min(1.0, excess / span)
            is_val *= (1.0 + GLUCOTOX_MAX_IS_INCREASE * intensity)

        # Postprandial IR: T1DM lacks the GLP-1 boost; absorbing carb is mildly insulin-resistant.
        if active_carb > 0.0:
            penalty = POSTPRANDIAL_IR_PENALTY_FACTOR * active_carb / (POSTPRANDIAL_IR_PENALTY_HALF + active_carb)
            is_val *= (1.0 + penalty)

        # Fast AR(1) noise: same stationary sigma as before, ~22min half-life for smooth IS swings.
        self._ar_is = (NOISE_AR1_RHO_METABOLIC * self._ar_is
                       + NOISE_AR1_INNOV_METABOLIC * self.noise_rng.normal(0, IS_FAST_NOISE_SIGMA))
        is_val *= (1.0 + self._ar_is)

        return max(0.2, is_val)

    def _compute_cgm_observation(self, true_bg: float) -> float:
        """CGM reading: interstitial lag then proportional noise. IG += (BG-IG)*(1-exp(-dt/tau)),
        tau=cgm_lag_minutes (Rebrin/Steil); sensor reports the delayed value plus AR(1) noise.
        """
        lag = self.patient.cgm_lag_minutes
        if lag > 0:
            alpha_lag = 1.0 - np.exp(-DT_MINUTES / float(lag))
            self._interstitial_bg += alpha_lag * (true_bg - self._interstitial_bg)
            sensed_bg = self._interstitial_bg
        else:
            sensed_bg = true_bg

        # AR(1) sensor noise mimics real CGM drift over 30-60min (rho=0.92, ~42min half-life).
        self._ar_cgm = (NOISE_AR1_RHO_SENSOR * self._ar_cgm
                        + NOISE_AR1_INNOV_SENSOR * self.noise_rng.normal(0, CGM_NOISE_FRACTION))
        observed = sensed_bg * (1.0 + self._ar_cgm)
        return np.clip(observed, BG_CLAMP_MIN, BG_CLAMP_MAX)

    def _check_and_correct(self, time_idx: int):
        """Patient checks CGM and possibly corrects highs/lows."""
        p = self.patient
        s = self.state

        # Severe hypo (<55) is unignorable; awake acts now, asleep wakes. Bypasses 6h+ danger.
        severe_hypo = s.bg_observed < SEVERE_HYPO_THRESHOLD

        is_awake = self._today_wake_idx <= time_idx < self._today_sleep_idx
        if not is_awake and not severe_hypo:
            return

        # Check interval — bypassed by severe hypo
        steps_since_check = time_idx - s.last_cgm_check_idx
        check_interval_steps = int(p.cgm_check_interval_min / DT_MINUTES)
        if steps_since_check < check_interval_steps and not severe_hypo:
            return

        s.last_cgm_check_idx = time_idx

        eff_low_thresh = p.hypo_threshold

        # --- Handle hypoglycemia ---
        if s.bg_observed < eff_low_thresh:
            # Refractory stops carb-stacking; severe uses a 10min gap so rage-eating still waits.
            refractory_min = SEVERE_HYPO_REFRACTORY_MIN if severe_hypo else HYPO_CORRECTION_REFRACTORY_MIN
            refractory_steps = int(refractory_min / DT_MINUTES)
            if time_idx - s.last_hypo_correction_idx < refractory_steps:
                return

            # Rule-of-15 recheck accounts for rescue glucose absorbing; rage-eat skips the math.
            rescue_cob = (float(np.sum(self._rescue_totals[time_idx:]))
                          if time_idx < len(self._rescue_totals) else 0.0)
            rage_eat = self.rng.random() < RAGE_EAT_PROBABILITY_BASE * (1.0 - p.dosing_competence)
            if rage_eat:
                cob_consideration = 0.0
            else:
                awareness = COB_AWARENESS_BASE + COB_AWARENESS_SKILL * p.dosing_competence
                cob_consideration = rescue_cob * BG_SCALE_FACTOR * awareness
            # Where the patient reckons they are heading once that glucose lands.
            projected_bg = s.bg_observed + cob_consideration
            if projected_bg >= eff_low_thresh:
                return  # already treated enough — wait for it to act

            # Grams lift the projection to threshold plus a margin; panic adds overshoot.
            deficit = eff_low_thresh + HYPO_RESCUE_TARGET_MARGIN - projected_bg
            correction_grams = max(HYPO_RESCUE_MIN_GRAMS, deficit / BG_SCALE_FACTOR * (
                HYPO_RESCUE_DEFICIT_GAIN + p.panic_factor * HYPO_RESCUE_PANIC_GAIN))

            # Hypo correction uses fast-acting carbs (glucose tablets / juice)
            k = HYPO_CARB_K
            theta = HYPO_CARB_THETA
            duration = max(k * theta * 4, 60)
            curve = gamma_curve(correction_grams, k, theta, duration)
            self.inject_curve(curve, time_idx, 'correction_carb',
                              f'Hypo correction {correction_grams:.0f}g')
            s.last_hypo_correction_idx = time_idx

            # Post-hypo basal scale-down (~90min) mirrors real pump-suspend/skip-basal after a hypo.

            # Without it, forward basal keeps clearing glucose as fast as eaten, sawtoothing snacks.
            suspend_steps = int(POST_HYPO_BASAL_SUSPEND_DURATION_HOURS * 60 / DT_MINUTES)
            # Each correction adds a fresh sin^2 envelope; overlaps combine via min, staying smooth.
            s.post_hypo_basal_suspend_windows.append(
                (time_idx, time_idx + suspend_steps))

            # After severe hypo, recheck soon vs full CGM interval; mild hypos keep normal cadence.
            if severe_hypo:
                recheck_steps = max(1, 15 // DT_MINUTES)
                s.last_cgm_check_idx = time_idx - check_interval_steps + recheck_steps

        # --- Trend-based anticipatory rescue carbs ---
        elif len(s.bg_obs_history) >= TREND_CORRECTION_WINDOW_STEPS:
            # Slope from CGM-observed history (bg_obs_history), not true BG: sensor-only action.
            window = s.bg_obs_history[-TREND_CORRECTION_WINDOW_STEPS:]
            trend = (window[-1] - window[0]) / (TREND_CORRECTION_WINDOW_STEPS - 1)
            if trend < TREND_LOW_RATE_THRESHOLD and s.bg_observed < TREND_LOW_BG_MAX:
                # Honors hypo-correction refractory: trend-low + regular correction don't stack.
                refractory_steps = int(HYPO_CORRECTION_REFRACTORY_MIN / DT_MINUTES)
                if time_idx - s.last_hypo_correction_idx < refractory_steps:
                    return
                if self.rng.random() < p.attentiveness:
                    correction_grams = float(np.clip(
                        abs(trend) * TREND_CORRECTION_WINDOW_STEPS * 2.0, 5.0, 20.0))
                    k = HYPO_CARB_K
                    theta = HYPO_CARB_THETA
                    duration = max(k * theta * 4, 60)
                    curve = gamma_curve(correction_grams, k, theta, duration)
                    self.inject_curve(curve, time_idx, 'correction_carb',
                                      f'Trend corr {correction_grams:.0f}g')
                    s.last_hypo_correction_idx = time_idx

    def generate(self) -> dict:
        """
        Generate one time step (DT_MINUTES).
        Returns a dict with all factor values and BG delta for this step.
        Like rand() in C - call repeatedly to advance.
        """
        idx = self.state.current_idx
        s = self.state
        p = self.patient

        # Check if we need to plan a new day
        if idx > 0 and idx % STEPS_PER_DAY == 0:
            s.day_number += 1
            self._plan_day()
            self._generate_day_events()

        # --- Activate pending events ---
        while self._pending_events and self._pending_events[0][0] <= idx:
            event_time, event_type, event_data = self._pending_events.pop(0)
            curve = event_data['curve']
            label = event_data.get('label', '')

            # Pre-bolus CGM glance: skip the bolus below the threshold, reduce it just above.
            if event_type == 'bolus':
                check_prob = BOLUS_BG_CHECK_BASE_PROB + 0.05 * p.attentiveness
                if self.rng.random() < check_prob:
                    bg = s.bg_observed
                    if bg < p.hypo_threshold:
                        continue
                    elif bg < p.hypo_threshold + BOLUS_REDUCE_MARGIN:
                        scale = BOLUS_REDUCE_FACTOR_BASE + 0.3 * p.dosing_competence
                        curve = curve * scale
                        label = f"{label} (low-BG reduced ×{scale:.2f})"

            # Same CGM glance before exercise, unconditional (deliberate act); skips IS boost too.
            elif event_type == 'exercise':
                if s.bg_observed < p.hypo_threshold + EXERCISE_HYPO_MARGIN:
                    continue

            self.inject_curve(curve, event_time, event_type, label=label)
            # Schedule post-exercise IS sensitivity boost
            if event_type == 'exercise':
                ex_dur = event_data.get('duration_min', EXERCISE_DURATION_MEAN_MIN)
                effect_start = event_time + int(ex_dur / DT_MINUTES)
                effect_end = effect_start + int(EXERCISE_IS_DURATION_HOURS * 60 / DT_MINUTES)
                reduction = min(0.30, EXERCISE_IS_REDUCTION * (ex_dur / EXERCISE_DURATION_MEAN_MIN))
                s.exercise_is_effects.append((effect_start, effect_end, reduction))

        # --- Read per-step contributions from pre-computed accumulation arrays (O(1)) ---
        total_carb = float(self._carb_totals[idx]) if idx < len(self._carb_totals) else 0.0
        rescue_carb = float(self._rescue_totals[idx]) if idx < len(self._rescue_totals) else 0.0
        basal_step = float(self._basal_totals[idx]) if idx < len(self._basal_totals) else 0.0
        bolus_step = float(self._bolus_totals[idx]) if idx < len(self._bolus_totals) else 0.0
        # Post-hypo basal stand-down (pump suspend/skip-basal); sin^2 envelopes combine via min.

        # Max slope pi/window: 4h window, depth 0.50 caps worst 10-min change ~6.5%; never jumps up.
        if s.post_hypo_basal_suspend_windows:
            min_factor = 1.0
            still_active = []
            for start, until in s.post_hypo_basal_suspend_windows:
                if idx >= until:
                    continue  # expired — drop it
                still_active.append((start, until))
                if idx < start:
                    continue
                total = until - start
                envelope = np.sin(np.pi * (idx - start) / total) ** 2
                factor = 1.0 - envelope * (1.0 - POST_HYPO_BASAL_SUSPEND_FACTOR)
                if factor < min_factor:
                    min_factor = factor
            s.post_hypo_basal_suspend_windows = still_active
            basal_step *= min_factor
        total_insulin = basal_step + bolus_step
        total_exercise = float(self._exercise_totals[idx]) if idx < len(self._exercise_totals) else 0.0

        # Per-step AR(1) noise (gut/depot don't reset every 5min); matches original sigma constants.
        self._ar_carb = (NOISE_AR1_RHO_METABOLIC * self._ar_carb
                         + NOISE_AR1_INNOV_METABOLIC * self.noise_rng.normal(0, CARB_ABSORPTION_NOISE_SIGMA))
        self._ar_insulin = (NOISE_AR1_RHO_METABOLIC * self._ar_insulin
                            + NOISE_AR1_INNOV_METABOLIC * self.noise_rng.normal(0, INSULIN_ABSORPTION_NOISE_SIGMA))
        # Absorption noise perturbs what reaches blood; recorded channels keep the declared curves.
        absorbed_carb = max(0.0, total_carb * (1.0 + self._ar_carb))
        absorbed_insulin = max(0.0, total_insulin * (1.0 + self._ar_insulin))

        # EMA-smoothed insulin feeds the Hill fn so HGO doesn't step when a bolus curve activates.
        self._smoothed_insulin_for_hgo = (
            HGO_INSULIN_SMOOTHING_ALPHA * absorbed_insulin
            + (1.0 - HGO_INSULIN_SMOOTHING_ALPHA) * self._smoothed_insulin_for_hgo
        )
        self._ar_hgo = (NOISE_AR1_RHO_METABOLIC * self._ar_hgo
                        + NOISE_AR1_INNOV_METABOLIC * self.noise_rng.normal(0, HGO_NOISE_SIGMA))
        hgo_rate = compute_hgo_rate(self._smoothed_insulin_for_hgo) * (1 + self._ar_hgo)
        hgo_value = hgo_rate * (DT_MINUTES / 60.0)
        # Scales HGO by body weight (heavier liver, more output); basal calibration mirrors this.
        hgo_value *= self.patient.body_weight_kg / BODY_WEIGHT_MEAN_KG

        # Circadian HGO: dawn surge (~6:30am) + deep-sleep trough (~3am), added as g/hr, not scaled.
        hour_of_day = (idx * DT_MINUTES / 60.0) % 24.0
        night_h = hour_of_day if hour_of_day < 12 else hour_of_day - 24
        dawn_g_per_hr = self.patient.dawn_hgo_amplitude * np.exp(
            -0.5 * ((hour_of_day - DAWN_HGO_PEAK_HOUR) / DAWN_HGO_SIGMA_HOURS) ** 2)
        night_dip_g_per_hr = self.patient.night_hgo_dip_amplitude * np.exp(
            -0.5 * ((night_h - NIGHT_HGO_DIP_HOUR) / NIGHT_HGO_DIP_SIGMA_HOURS) ** 2)
        hgo_value += (dawn_g_per_hr - night_dip_g_per_hr) * (DT_MINUTES / 60.0)
        hgo_value = max(0.0, hgo_value)

        # Glycogen gating: when low, glycogenolysis can't sustain HGO; drops toward gluconeo floor.
        glycogen_low_threshold = GLYCOGEN_CAPACITY_GRAMS * GLYCOGEN_LOW_THRESHOLD_FRACTION
        if s.glycogen_grams < glycogen_low_threshold:
            # Linear scaling on the glycogenolysis-sourced fraction of HGO
            availability = max(0.0, s.glycogen_grams / glycogen_low_threshold)
            hgo_value *= (1.0 - GLYCOGEN_DRAIN_FRACTION) + GLYCOGEN_DRAIN_FRACTION * availability

        # Alcohol suppresses HGO (gluconeogenesis block); trapezoid avoids a step at session edges.
        alc_ramp_steps = int(ALCOHOL_HGO_RAMP_HOURS * 60 / DT_MINUTES)
        alcohol_hgo_factor = 1.0
        active_alcohol = []
        for (start_idx, end_idx, hgo_factor) in s.alcohol_effects:
            if idx < end_idx:
                active_alcohol.append((start_idx, end_idx, hgo_factor))
                intensity = envelope_intensity(idx, start_idx, end_idx,
                                                alc_ramp_steps, alc_ramp_steps)
                if intensity > 0:
                    eff_factor = 1.0 + (hgo_factor - 1.0) * intensity
                    alcohol_hgo_factor = min(alcohol_hgo_factor, eff_factor)
        s.alcohol_effects = active_alcohol
        hgo_value *= alcohol_hgo_factor

        # Delayed HGO rebound: large meals bump HGO 3.5-5.5h later; trapezoid envelope, no effect=0.
        meal_hgo_ramp_steps = int(DELAYED_HGO_RAMP_HOURS * 60 / DT_MINUTES)
        meal_hgo_bump = 0.0
        active_meal_hgo = []
        for (start_idx, end_idx, magnitude) in s.meal_hgo_effects:
            if idx < end_idx:
                active_meal_hgo.append((start_idx, end_idx, magnitude))
                intensity = envelope_intensity(idx, start_idx, end_idx,
                                                meal_hgo_ramp_steps, meal_hgo_ramp_steps)
                if intensity > 0:
                    meal_hgo_bump += magnitude * intensity
        s.meal_hgo_effects = active_meal_hgo
        if meal_hgo_bump > 0:
            hgo_value += meal_hgo_bump * (DT_MINUTES / 60.0)

        # Glycogen drains by the glycogenolysis fraction of HGO, refills from carbs (background).

        # Doesn't subtract from BG-bound carbs (ICR tuned to net response); gating couples it to BG.
        s.glycogen_grams -= hgo_value * GLYCOGEN_DRAIN_FRACTION
        s.glycogen_grams += absorbed_carb * GLYCOGEN_REFILL_FRACTION
        s.glycogen_grams = float(np.clip(s.glycogen_grams, 0.0, GLYCOGEN_CAPACITY_GRAMS))

        # Remove expired entries from active_curves (memory management for external consumers)
        s.active_curves = [c for c in s.active_curves
                           if (idx - c.start_time_idx) < len(c.values)]

        # --- Insulin sensitivity (modulates insulin effectiveness, not carb load) ---
        insulin_resistance_factor = self._compute_insulin_resistance(idx, active_carb=absorbed_carb)

        # IS divides insulin's effect: resistant (IR>1) clears less; sensitive (IR<1) clears more.
        glucose_in = absorbed_carb + hgo_value - total_exercise
        glucose_out = absorbed_insulin * p.icr / insulin_resistance_factor
        bg_delta = BG_SCALE_FACTOR * (glucose_in - glucose_out)

        # Sg: insulin-independent pull to a stochastic OU equilibrium (reverts to anchor+dawn lift).

        # E timescale (not Sg strength) keeps 8h ACF near zero; Sg stays weak, high-passes less.
        ge_mu = p.ge_anchor + p.ge_dawn_amplitude * ge_diurnal_profile(hour_of_day)
        ge_rho = float(np.exp(-DT_MINUTES / (GE_EQ_TAU_HOURS * 60.0)))
        self._ge_equilibrium = (
            ge_mu + ge_rho * (self._ge_equilibrium - ge_mu)
            + np.sqrt(1.0 - ge_rho * ge_rho) * GE_EQ_SIGMA * p.ge_sigma_mult * self.noise_rng.normal())
        self._ge_equilibrium = max(self._ge_equilibrium, GE_EQ_FLOOR)
        bg_delta += p.glucose_effectiveness * (self._ge_equilibrium - s.bg)

        # Physiological guardrails
        if s.bg > RENAL_THRESHOLD:
            bg_delta -= (s.bg - RENAL_THRESHOLD) * RENAL_CLEARANCE_RATE

        if s.bg < COUNTER_REGULATORY_THRESHOLD:
            bg_delta += COUNTER_REGULATORY_RATE * (COUNTER_REGULATORY_THRESHOLD - s.bg) / COUNTER_REGULATORY_THRESHOLD

        # Severe-hypo glucagon dump — escalates the response below SEVERE_HYPO_THRESHOLD
        if s.bg < SEVERE_HYPO_THRESHOLD:
            severity = (SEVERE_HYPO_THRESHOLD - s.bg) / SEVERE_HYPO_THRESHOLD
            bg_delta += SEVERE_HYPO_GLUCAGON_RATE * severity

        # Soft ceiling measures headroom from the higher of BG and the soft edge: monotone in dose.
        ceiling_ref = max(s.bg, BG_SOFT_CEILING)
        highest = ceiling_ref + SOFT_APPROACH_FRACTION * (BG_CLAMP_MAX - ceiling_ref)
        bg_delta = min(bg_delta, highest - s.bg)

        # True BG has no floor; only the CGM reading is clipped to BG_CLAMP_MIN.
        s.bg = float(min(s.bg + bg_delta, BG_CLAMP_MAX))

        # Glucotox BG EMA (~3h half-life) drives transient IR when chronically elevated.
        glucotox_alpha = 1.0 - 0.5 ** (DT_MINUTES / (GLUCOTOX_BG_EMA_HALF_LIFE_HOURS * 60.0))
        s.glucotox_bg_ema = glucotox_alpha * s.bg + (1.0 - glucotox_alpha) * s.glucotox_bg_ema

        # CGM observation
        s.bg_observed = self._compute_cgm_observation(s.bg)

        # --- Patient behavior (check and correct) ---
        self._check_and_correct(idx)

        # --- Record history ---
        s.bg_history.append(s.bg)
        s.bg_obs_history.append(s.bg_observed)
        s.carb_curve_history.append(total_carb)
        s.insulin_curve_history.append(total_insulin)
        s.resistance_history.append(insulin_resistance_factor)
        s.exercise_curve_history.append(total_exercise)
        s.hgo_history.append(hgo_value)
        s.delta_history.append(bg_delta)

        # Advance
        s.current_idx = idx + 1

        time_hours = (idx * DT_MINUTES) / 60.0
        day = int(time_hours / 24)
        hour_of_day = time_hours % 24.0

        return {
            'index': idx,
            'time_hours': time_hours,
            'day': day,
            'hour_of_day': hour_of_day,
            'bg': s.bg,
            'bg_observed': s.bg_observed,
            'bg_delta': bg_delta,
            'total_carb': total_carb,
            'rescue_carb': rescue_carb,
            'total_insulin': total_insulin,
            'basal_insulin': basal_step,
            'bolus_insulin': bolus_step,
            'total_exercise': total_exercise,
            'insulin_resistance': insulin_resistance_factor,
            'hgo': hgo_value,
            'glucose_in': glucose_in,
            'glucose_out': glucose_out,
            'is_sick': s.is_sick,
            'is_rare_day': s.is_rare_event_day,
            'is_weekend': s.day_of_week >= 5,
            'is_holiday': s.is_holiday,
            'alcohol_hgo_factor': alcohol_hgo_factor,
        }

    def generate_hours(self, hours: float) -> dict:
        """Generate multiple steps at once. Returns dict of numpy arrays."""
        n_steps = int(hours * 60 / DT_MINUTES)
        results: dict = {
            'index': [], 'time_hours': [], 'day': [], 'hour_of_day': [],
            'bg': [], 'bg_observed': [], 'bg_delta': [],
            'total_carb': [], 'rescue_carb': [], 'total_insulin': [], 'basal_insulin': [],
            'bolus_insulin': [], 'total_exercise': [],
            'insulin_resistance': [], 'hgo': [], 'glucose_in': [], 'glucose_out': [],
            'is_sick': [], 'is_rare_day': [], 'is_weekend': [], 'is_holiday': [],
            'alcohol_hgo_factor': [],
        }
        for _ in range(n_steps):
            step = self.generate()
            for k, v in step.items():
                results[k].append(v)

        return {k: np.array(v) for k, v in results.items()}

    def get_patient_summary(self) -> dict:
        """Return a summary of the patient's profile."""
        p = self.patient
        return {
            'dietary_discipline': f'{p.dietary_discipline:.3f}',
            'attentiveness': f'{p.attentiveness:.3f}',
            'dosing_competence': f'{p.dosing_competence:.3f}',
            'lifestyle_consistency': f'{p.lifestyle_consistency:.3f}',
            'is_base': f'{p.is_base:.2f}',
            'icr': f'{p.icr:.1f}',
            'correction_factor': f'{p.correction_factor:.1f}',
            'basal_dose': f'{p.basal_dose:.1f}U',
            'basal_duration': f'{p.basal_duration_hours:.1f}h',
            'cgm_check_interval': f'{p.cgm_check_interval_min:.0f}min',
            'bolus_per_day': f'{BOLUS_NIGHT_EVENTS_PER_DAY + BOLUS_DAY_EVENTS_PER_DAY:.1f}',
            'exercise_prob': f'{p.exercise_probability:.2f}',
            'basal_miss_prob': f'{p.basal_miss_prob:.4f}',
            'slow_carb_pref': f'{p.slow_carb_preference:.2f}',
            'panic_factor': f'{p.panic_factor:.2f}',
        }


# QUICK TEST

if __name__ == '__main__':
    sim = T1DMSimulator(seed=42)
    print("Patient:", sim.get_patient_summary())
    data = sim.generate_hours(24)
    print(f"24h BG range: {data['bg'].min():.0f} - {data['bg'].max():.0f} mg/dL")
    print(f"Mean BG: {data['bg'].mean():.0f} mg/dL")
    print(f"Steps: {len(data['bg'])}")
