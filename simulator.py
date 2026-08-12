"""
T1DM Patient Behavior Simulator
================================
Generates synthetic factor curves (carb intake, insulin, sensitivity, exercise)
and computes blood sugar deltas. Designed for training transformer models on
patient behavior patterns.

Architecture:
- Patient profile is sampled from a multivariate normal (4 skill dimensions)
- All behavioral parameters are derived from the skill profile
- Output is factor curves + BG trace at 5-minute resolution
- Seed-driven PRNG for reproducibility
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

# ============================================================================
# GLOBAL SIMULATION PARAMETERS (tweak these freely)
# ============================================================================

# Time resolution
DT_MINUTES = 5  # Time step in minutes
STEPS_PER_DAY = 24 * 60 // DT_MINUTES  # 288 steps per day

# Hours a fresh simulator must run before its state is at steady state, and the
# canonical value every consumer discards. The basal pipeline starts empty (first
# events land 2-7 h in) and Bateman PK does not settle until ~36-48 h given the
# 18-30 h per-patient cadence; a shorter lead-in leaves the trace in a cold-start
# ramp that peaks near 280 mg/dL. Whole days, so day-of-week arithmetic stays exact.
SIMULATOR_WARMUP_HOURS = 48.0

# Skill correlation
SKILL_CORRELATION = 0.7  # Off-diagonal correlation in skill covariance matrix
SKILL_VARIANCE = 0.6  # [HIVAR 2x] 0.3→0.6 — wider skill spread
SKILL_MIN = 0.15  # [HIVAR 2x] 0.25→0.15 — low-skill floor
SKILL_MAX = 0.98  # [HIVAR 2x] 0.95→0.98 — high-skill ceiling

# Wake/sleep
WAKE_TIME_MEAN_HOURS = 7.5  # Mean wake time (hours from midnight). At 6.5 the dawn HGO
                            # surge + breakfast absorption together peaked at hour 6-7,
                            # while real cohorts (Ohio, Shanghai) peak at hour 8. 7.5 puts
                            # breakfast at ~08:00 so the breakfast-absorption peak lands at
                            # ~09:30, stacking with the (now later) dawn HGO peak to put
                            # the diurnal maximum at hour 8 rather than hour 6-7.
WAKE_TIME_SIGMA_BASE = 1.0  # [HIVAR 2x] 0.5→1.0 — wake-time scatter
SLEEP_DURATION_MEAN_HOURS = 7.5
SLEEP_DURATION_SIGMA_HOURS = 2.0  # [HIVAR 2x] 1.0→2.0 — sleep-length scatter

# Meals
MEALS_BASE = 4  # [GE-FIX] 3→4 — added a mid-afternoon eating occasion (see MEAL_TIME_OFFSETS_HOURS) to fill the lunch->dinner gap that left the afternoon BG sagging ~15 mg/dL below Ohio; heavy meal-timing jitter keeps it aperiodic (real forcing, not a deterministic daily bump)
MEALS_EXTRA_LAMBDA = 3.5  # Extra meals Poisson lambda, scaled by (1 - s1). High value provides sub-day stochastic shocks that decorrelate the BG trace at mid-range lags.
MEAL_TIME_OFFSETS_HOURS = [0.5, 5.0, 8.5, 12.0]  # Breakfast, lunch, afternoon snack, dinner (offset from wake); the wake+8.5 (~16:00) slot fills the old lunch->dinner gap.
                                             # Dinner at wake+12 = 19:30 for the 7:30 wake
                                             # population — the physiologic norm. Earlier
                                             # values (10.0) placed dinner at 16:30, ending
                                             # all carb absorption by ~18:00 and leaving the
                                             # 18:00-00:00 window covered only by basal,
                                             # which over-cleared into a -24 mg/dL post-
                                             # dinner crash between hour 21 and midnight.
MEAL_TIME_JITTER_BASE_MIN = 120.0  # [GE-FIX] 30→120 — real T1D meal times scatter by hours; this aperiodic timing spread decorrelates the deterministic daily BG hump that otherwise dominated the 4-12h autocorrelation once the trace was tightened by Sg
MEAL_CARB_MEANS = [32.0, 42.0, 20.0, 35.0]  # breakfast, lunch, afternoon snack (20g), dinner. Dinner cut to 35g:
                                       # the prior [48, 63, 75] over-shot Ohio's evening BG by
                                       # +50 mg/dL at hour 21 and the morning BG by +27 at hour
                                       # 10. Real T1Ds in published cohorts cluster nearer
                                       # 40-60g for the main meals. Daily total of ~155 g still
                                       # within Ohio's range once snacks/weekend jitter add ~30g.
MEAL_CARB_SIGMA = 16.0  # [OHIO-CARB] 44→16 — per-meal size spread. The prior 44 with the max(0,·) floor truncated a wide Gaussian at zero, which inflated every meal's realised mean far above target (the population ate ~273 g/day vs Ohio's ~194). A realistic per-meal SD that barely clips typical 20-42g meals; between-patient spread now comes from meal_appetite, not per-meal noise.
MEAL_CARB_DISCIPLINE_SCALE = 0.7  # How much s1 reduces carb intake
# [OHIO-CARB] Per-patient multiplicative appetite: a lognormal "how much this
# person eats" trait, orthogonal to skill (discipline already trims carbs via
# MEAL_CARB_DISCIPLINE_SCALE). Lognormal gives median 1.0 with a right tail, so
# the population carbs/day picks up Ohio's right skew (mean 194 > median 163) and
# its wide between-patient spread (sd ~90, range ~90-373) that per-meal noise
# cannot supply. Applied to carb AMOUNT only, never meal count/timing, so the
# ACF-relevant meal shocks are untouched.
MEAL_APPETITE_LOG_SIGMA = 0.45  # [OHIO-CARB] between-patient appetite spread (lognormal sigma) — tuned so pooled carbs/day sd ~90 g matches Ohio
MEAL_APPETITE_CLIP = (0.45, 2.05)  # [OHIO-CARB] clip the lognormal tails (smallest/largest eaters ~90 / ~370+ g/day)
MEAL_CARB_SCALE = 1.08  # [OHIO-CARB] global fine-tune on all meal carb amounts to land the population mean on Ohio (~194 g/day)
SNACK_CARB_MEAN = 20.0
SNACK_CARB_SIGMA = 20.0  # [HIVAR 2x] 10.0→20.0 — unused by generation

# Protein/fat baseline — peak around 75 min (k=4, θ=25). Earlier value (k=3.5)
# regressed mean BG; original k=6 dragged the cohort envelope peak past 200 min.
# This shape sits the protein/fat tail close to the OhioT1DM cohort post-meal
# peak time of ~75-100 min.
PROTEIN_FAT_GAMMA_K = 4.0
PROTEIN_FAT_GAMMA_THETA = 25.0
PROTEIN_FAT_FRACTION_OF_CARBS = 0.24
PROTEIN_FAT_MIN_GRAMS = 6.0
PROTEIN_FAT_MAX_GRAMS = 18.0

# Carb curve (gamma distribution parameters)
# Peak time = (k-1)*theta. Mean = k*theta.
# FAST_CARB_K/THETA are reference values used by the test suite to construct
# benchmark curves. Production meal generation samples k/theta from the
# MIXED_MEAL_FAST_*_RANGE constants instead — these are NOT the live values.
FAST_CARB_K = 3.0  # Gamma shape for fast carbs (peak ~40 min)
FAST_CARB_THETA = 20.0  # Gamma scale for fast carbs (minutes)
# slow_carb_preference of ~0.18 for mid-skill keeps the cohort post-meal
# envelope peak close to the ~100 min observed in OhioT1DM. Higher values
# (e.g. 0.55) put 47% of meal carbs on the late slow tail, dragging the peak
# past 200 min.
SLOW_CARB_PREFERENCE_BASE = 0.10  # Base probability of choosing slow carbs
SLOW_CARB_PREFERENCE_SKILL_BONUS = 0.15  # Added probability from s1

# Hypo correction carbs (glucose tablets / juice — kick in faster than meal carbs)
HYPO_CARB_K = 2.0
HYPO_CARB_THETA = 15.0  # Peak ~15 min

# Slow-tail follow-up snack after a hypo correction (clinical "rule-of-15 plus snack").
# Damps the recurrent dip 60-90 min later when fast carbs are gone but the meal bolus is
# still acting. Skill-gated: only attentive patients remember the follow-up.
HYPO_FOLLOWUP_FRACTION = 0.30      # Fraction of rescue dose, delivered as slow carbs. Longer slow-carb tail keeps BG above 70 through the next hour, blocking re-hypo.
HYPO_FOLLOWUP_GAMMA_K = 4.0        # Slow gamma — peaks around 90 min
HYPO_FOLLOWUP_GAMMA_THETA = 30.0   # Tail extends ~5h
HYPO_FOLLOWUP_SKILL_THRESHOLD = 0.30  # Most patients eat the follow-up; only the very lowest-skill skip it.

# Carb curve noise
CARB_CURVE_K_NOISE = 0.2  # [HIVAR 2x] 0.1→0.2 — absorption-shape scatter
CARB_CURVE_THETA_NOISE = 0.2  # [HIVAR 2x] 0.1→0.2 — absorption-shape scatter

# Mixed-meal composition (each meal becomes 2-5 overlapping carb components)
MIXED_MEAL_MIN_COMPONENTS = 2
MIXED_MEAL_EXTRA_COMPONENTS_LAMBDA = 1.5  # Poisson, added to MIN
MIXED_MEAL_MAX_COMPONENTS = 5
MIXED_MEAL_DIRICHLET_ALPHA = 1.5  # Higher = more uniform fractions per component
MIXED_MEAL_FAST_K_RANGE = (2.0, 3.5)
MIXED_MEAL_FAST_THETA_RANGE = (15.0, 22.0)
MIXED_MEAL_MED_K_RANGE = (3.0, 4.5)
MIXED_MEAL_MED_THETA_RANGE = (20.0, 28.0)
# Slow components shifted earlier to bring the cohort post-meal envelope peak
# from ~200 min back toward the ~100 min seen in OhioT1DM. Original (4-6,28-45)
# put peaks at 84-225 min; the new ranges put them at 55-140 min while still
# being the latest of the three categories. Tighter (3.0-4.5) dropped mean BG
# below the ±3 threshold so this is the working compromise.
MIXED_MEAL_SLOW_K_RANGE = (3.5, 5.0)
MIXED_MEAL_SLOW_THETA_RANGE = (22.0, 35.0)
MIXED_MEAL_MED_WEIGHT_BASE = 0.4  # Base weight for medium-speed components

# Body weight and insulin resistance — two per-patient axes. Real T1D
# populations include thin sensitive patients (low TDD ~25 U/day) and heavy
# IR patients (TDD >100 U/day); both ends need to be reachable.
#
# body_weight_kg scales HGO (heavier liver → more endogenous glucose) and via
# the basal-balance rule, basal dose. insulin_resistance_factor scales ICR
# inversely (resistant patients need more insulin per gram of carbs) AND sets
# is_base (resistance reduces glucose clearance per unit insulin). The two
# axes are independent so a thin-IR patient and an obese-sensitive patient
# both exist.
BODY_WEIGHT_MEAN_KG = 75.0
BODY_WEIGHT_SIGMA_KG = 36.0  # [HIVAR 2x] 18.0→36.0 — body-weight spread
BODY_WEIGHT_MIN_KG = 42.0  # [HIVAR 2x] 45.0→42.0 — light-weight floor
BODY_WEIGHT_MAX_KG = 150.0  # [HIVAR 2x] 130.0→150.0 — heavy-weight ceiling
IR_LOGNORMAL_SIGMA = 0.26  # [HIVAR 2x][HET] 0.08→0.16→0.26 — insulin-resistance spread. Widened (with a stronger IR→anchor coupling below) to carry the between-patient MEAN-BG spread: the OU anchor sigma alone saturates (~14 mg/dL sd) because the GE_EQ_FLOOR compresses low-anchor patients upward, whereas a wider IR axis spreads patients on the HIGH side (resistant → higher anchor via the coupling, higher ICR/CF), which the floor does not compress and which does not crash into hypo.
IR_FACTOR_MIN = 0.4  # [HIVAR 2x] 0.5→0.4 — sensitive end
IR_FACTOR_MAX = 2.0  # [HIVAR 2x] 1.5→2.0 — resistant end
IR_TO_IS_NOISE_SIGMA = 0.2  # [HIVAR 2x] 0.1→0.2 — is_base decoupling
IR_TO_ICR_NOISE_SIGMA = 0.24  # [HIVAR 2x] 0.12→0.24 — ICR decoupling

# Insulin sensitivity. is_base is derived from insulin_resistance_factor
# rather than sampled independently; this constant stays as the reference
# centerpoint for the lognormal draw and isn't directly used in generation.
IS_BASE_MEAN = 1.0
IS_BASE_SIGMA = 0.2
IS_DAILY_DRIFT_SIGMA = 0.2  # [HIVAR 2x] 0.1→0.2 — day-to-day sensitivity drift
IS_FAST_NOISE_SIGMA = 0.0  # [MINNOISE] step-to-step insulin-resistance noise minimized to zero
IS_DAWN_PHASE_DAILY_SIGMA = 3.0  # [HIVAR 2x] 1.5→3.0 — dawn-timing scatter
IS_DRIFT_TRANSITION_HOURS = 4.0  # Smooth blend across midnight from prev to today's drift/phase

# Insulin sensitivity diurnal components (multiple peaks). All amplitudes are
# deviations from a baseline IS of 1.0. Real CGM shows a strong dawn rise
# peaking ~8am driven mainly by cortisol-mediated IR + dawn HGO surge; a
# milder evening cortisol peak; and lowest BG around 11pm-2am during deep
# sleep when IS is highest.
IS_MORNING_PEAK_HOUR = 7.5    # Morning resistance peak
IS_MORNING_AMPLITUDE = 0.05   # Cut from 0.30: at the old value, insulin was 30% less
                              # effective during the breakfast-absorption window, which
                              # stacked with the widened dawn HGO surge to drive a +37 mg/dL
                              # morning-peak overshoot vs Ohio. Real morning cortisol-driven
                              # IR is closer to 5-15%; 0.30 was a relic of when the model
                              # needed exaggerated morning IR to compensate for absent
                              # dawn-HGO. With the dawn HGO surge now doing that work,
                              # IS_MORNING_AMPLITUDE can return to a physiologic baseline.
IS_EVENING_PEAK_HOUR = 20.0   # Evening resistance peak
IS_EVENING_AMPLITUDE = 0.0    # Zeroed. Real cohorts show no evening BG bulge; the previous
                              # 0.08 made insulin 8% less effective at hour 20, slowing dinner
                              # clearance precisely when we needed it fastest.
IS_NIGHT_DIP_HOUR = 2.0       # Nighttime sensitivity peak (low resistance)
IS_NIGHT_DIP_AMPLITUDE = 0.15 # How much more sensitive at night

# Illness
ILLNESS_PROBABILITY_BASE = 0.06  # Per-day probability of getting sick
ILLNESS_HEALTH_WEIGHT = 0.8  # How much s4 reduces illness probability
ILLNESS_RECOVERY_PROB = 0.2  # Geometric distribution parameter
ILLNESS_IS_FACTOR_MIN = 1.25  # [HIVAR 2x] 1.3→1.25 — illness IR range
ILLNESS_IS_FACTOR_MAX = 3.0  # [HIVAR 2x] 2.5→3.0 — illness IR range
ILLNESS_IS_RAMP_RATE = 0.4  # How fast illness IS factor changes per day (0 to 1)

# Basal insulin (long-acting)
# Note: ideal basal dose is derived from HGO and ICR in generate_patient().
BASAL_DOSE_SIGMA = 9.0  # [HIVAR 2x] 4.5→9.0 — inter-patient basal-dose scatter
BASAL_DOSE_COMPETENCE_NOISE = 0.1  # [HIVAR 2x] 0.05→0.1 — day-to-day basal-dose noise
                                    # Tightened from 0.15: real intra-individual basal CV is 5-15%
                                    # for modern long-acting analogues (glargine ~12%, degludec ~6%),
                                    # not 20-30%. Larger values produced visible "uneven hill" basal
                                    # traces from dose-to-dose magnitude swings.
BASAL_DURATION_HOURS = 28.0  # Population reference duration of action (kept for
                              # tests and as the basal_curve default). Per-patient
                              # duration of action is set from the assigned long-acting
                              # analogue in BASAL_VARIANTS (glargine 26h / degludec 42h),
                              # not sampled from a continuous range; the injection cadence
                              # is a fixed once-daily BASAL_DOSE_INTERVAL_HOURS (24h),
                              # decoupled from the duration.
BASAL_DURATION_HOURS_MIN = 15.0  # Legacy basal-duration span bound — no longer read
BASAL_DURATION_HOURS_MAX = 34.0  # Legacy basal-duration span bound — no longer read
BASAL_MISS_PROB_BASE = 0.02  # Base probability of fully skipping a basal dose. Lowered from
                              # 0.10: at the legacy rate even high-skill patients missed ~1%, and
                              # low-skill patients missed ~35%, producing long full-cadence zero
                              # gaps in the basal trace. Real T1D MDI patients fully skip <3% of
                              # long-acting doses; "missed" doses are usually delayed by a few
                              # hours, not skipped (the PK overlap below absorbs short delays).
BASAL_MISS_SKILL_SCALE = 5.0  # How much skills reduce miss probability
BASAL_CORRECTION_MAX_ADJUSTMENT = 0.22  # Max % a patient will adjust basal vs base dose in one day.
                                        # Set wide enough that under-dosed IR patients can break out of
                                        # stuck-high streaks (lower caps left mean BG > 200 mg/dL for
                                        # weeks at a time even averaging over 7 days).
BASAL_KA_PER_HOUR = 0.30  # Absorption rate (1/h). Governs onset/time-to-peak.
                          # tmax = ln(ka/ke)/(ka-ke) ≈ 6.3h for ka=0.30, ke=0.07 —
                          # broader-peak long-acting profile sitting between glargine
                          # (peak ~4h) and degludec (peak ~9h). The flatter shape is
                          # required so consecutive once-daily doses (24h cadence)
                          # produce a near-flat cumulative basal trace rather than a
                          # peak-and-trough hill pattern.
BASAL_KE_PER_HOUR = 0.07  # Elimination rate (1/h). Half-life ≈ 9.9h; produces a
                          # broad peak and slow decline so the unit dose stays
                          # within ~50% of peak for the full cadence window.
BASAL_TAIL_CLIP_HOURS = 5.0  # Smootherstep window at the end of the curve that
                             # tapers the late residual to zero, so consecutive
                             # daily doses join without a tail-step discontinuity.
# Legacy PK-overlap fraction — no longer read. Each scheduled basal dose now
# generates a curve of duration = patient.basal_duration_hours (the analogue's
# action_hours), dosed at a fixed once-daily BASAL_DOSE_INTERVAL_HOURS (24h).
# Because action_hours (26h/42h) exceeds the 24h cadence, the previous dose's
# tail still overlaps the next, which supplies overnight coverage; degludec's
# long tail also bridges a skipped dose, glargine's shorter one largely does not.
BASAL_PK_OVERLAP_FRACTION = 1.00

# Per-basal-dose damping of the lipohypertrophy site-quality multiplier.
# `_site_quality` returns a per-injection absorption factor whose variance
# is calibrated for rapid-acting bolus into abdomen (the dominant rotation
# site). Long-acting basal goes into larger, less-frequently-used depots
# (thigh, buttock) and has much more consistent dose-to-dose absorption.
# We contract the per-basal site multiplier toward 1.0 by this factor so
# the basal trace doesn't inherit bolus-grade per-dose noise.
BASAL_SITE_QUALITY_DAMPING = 0.30
# Legacy aliases kept for tests / warmup math. With the Bateman PK these no
# longer represent literal trapezoid ramps — interpret as "approx time-to-peak"
# and "tail-clip duration" respectively.
BASAL_RAMP_UP_HOURS = 4.0
BASAL_RAMP_DOWN_HOURS = BASAL_TAIL_CLIP_HOURS

# Bolus insulin (rapid-acting)
# Duration of action scales with dose: BASE + SCALE * (sqrt(dose) - sqrt(5)).
# A 5U bolus uses BASE; a 1U bolus is shorter, a 20U bolus is longer.
# Theta also drifts with dose so larger boluses peak slightly later.
BOLUS_GAMMA_K = 3.0
BOLUS_GAMMA_THETA = 25.0  # Peak around 50 min for typical 5U dose
BOLUS_DURATION_HOURS = 4.0  # Legacy typical duration; new code uses bolus_pk_for_dose()
BOLUS_DIA_BASE_HOURS = 2.5  # Duration at the 5U reference dose. Shortened from 4.0:
                            # rapid-acting analogs (lispro, aspart, glulisine) have effective
                            # DIA of 3-4h. At 4.0 a typical 7U dinner bolus finished at 23:15,
                            # putting bolus_insulin >0.6 U/hr into the 22:00-02:00 window
                            # that drove the nocturnal BG crash on >65% of day-records.
                            # 3.0 lands the same bolus's tail at 22:15, exiting the nocturnal
                            # window. Combined with the dawn-HGO timing shift below, this is
                            # the structural attack on overnight clearance imbalance that
                            # IS-amplitude and basal-headroom dials could not reach.
BOLUS_DIA_DOSE_SCALE = 0.6  # Hours added per unit of sqrt(dose) - sqrt(5)
BOLUS_DIA_MIN_HOURS = 2.0  # Lowered from 3.0 so BOLUS_DIA_BASE_HOURS=2.5 is not clamped away.
BOLUS_DIA_MAX_HOURS = 7.5
BOLUS_THETA_DOSE_SLOPE = 0.06  # Theta multiplier per unit of sqrt(dose) - sqrt(5)
ICR_MEAN = 8.0   # Insulin-to-carb ratio (1 unit per X grams). Lowered from 11.0 → 8.0 to
                 # raise TDD into the clinical 0.5-0.7 U/kg/day band. At 11 the sim ran at
                 # 0.45 U/kg/day with a 55/45 basal:bolus split; 8 brings it to ~0.65 U/kg
                 # with a ~50/50 split that matches AZT1D's pump-log totals. ICR also enters
                 # `ideal_basal = HGO*24*is/ICR`, so lowering it raises both bolus and basal
                 # symmetrically while preserving the HGO-balances-basal invariant. The Hill
                 # calibration anchor `compute_hgo_rate(0.086) ≈ 8.25` is unaffected because
                 # it depends only on HGO_INSULIN_HALF_MAX and HGO_BASE, not ICR.
ICR_SIGMA = 2.0
BOLUS_TIMING_COMPETENT_MEAN = -15.0 # Minutes before meal (negative = before). Deepened from -5
                                     # to -15 so the bolus is already active when breakfast
                                     # carbs hit. With the shorter BOLUS_DIA_BASE_HOURS=2.5
                                     # and earlier wake (06:30), a 5-min pre-bolus left the
                                     # insulin curve trailing breakfast absorption by ~20 min,
                                     # contributing to the morning hyper. A
# small pre-bolus matches OhioT1DM behavior. Larger pre-boluses (e.g. -20 min)
# caused the cohort-aligned post-meal envelope to dip below baseline before
# rising, which is not seen in real CGM data.
BOLUS_TIMING_INCOMPETENT_MEAN = 10.0  # Minutes after meal
BOLUS_TIMING_SIGMA_BASE = 10.0  # [HIVAR 2x] 5.0→10.0 — bolus-timing scatter

# ---------------------------------------------------------------------------
# Real-world insulin analogues. Each patient is assigned exactly one
# rapid-acting bolus analogue and one long-acting basal analogue in
# generate_patient(). PK/PD parameters are sourced from prescribing
# information and published pharmacology reviews (cited inline). The
# glucose-lowering *action* profile — not the plasma concentration — is what
# the curves represent, since the curve drives glucose_out over time.
#
# Bolus analogues (rapid-acting) → gamma_curve; action peak = (k-1)*theta min,
# dia_base_hours = duration of action at the 5U reference dose (the dose
# scaling in bolus_pk_for_dose via BOLUS_DIA_*/BOLUS_THETA_DOSE_SLOPE is kept).
#   aspart  (NovoLog/NovoRapid): onset ~0.25h, peak ~1h,   DIA ~4h.
#       [StatPearls NBK500030; Fast-acting aspart review, PMC7007438]
#   lispro  (Humalog):           onset 0.25-0.5h, peak 1-2h, DIA ~5h.
#       [StatPearls NBK507840]
BOLUS_VARIANTS = {
    "aspart": {"gamma_k": 3.0, "gamma_theta": 30.0, "dia_base_hours": 4.0},  # peak ~60 min
    "lispro": {"gamma_k": 3.0, "gamma_theta": 37.5, "dia_base_hours": 5.0},  # peak ~75 min
}

# Basal analogues (long-acting) → basal_curve (Bateman one-compartment PK,
# f(t)=exp(-ke t)-exp(-ka t)); action peak tmax = ln(ka/ke)/(ka-ke). Both are
# dosed once daily (BASAL_DOSE_INTERVAL_HOURS cadence); the per-analogue
# action_hours sets the steady-state overlap and hence the flatness.
#   glargine U100 (Lantus): half-life ~12h (ke≈0.058/h), duration ~24h,
#       effective mainly over the first 12-18h; here tmax≈6.8h, action 26h.
#       [BEGIN Once Long, Diabetes Care 35:2464; glargine PI]
#   degludec (Tresiba): half-life >25h (ke≈0.028/h), duration >42h, plasma
#       peak 10-12h; here tmax≈11.5h (matches the sourced peak), action 42h.
#       [Clin Pharmacokinet review, PMC4156782]
BASAL_VARIANTS = {
    "glargine": {"ka": 0.30, "ke": 0.058, "action_hours": 26.0, "tail_clip_hours": 4.0},
    "degludec": {"ka": 0.20, "ke": 0.028, "action_hours": 42.0, "tail_clip_hours": 6.0},
}
BASAL_DOSE_INTERVAL_HOURS = 24.0  # Both analogues are injected once daily.

# Carb counting error — relative SD around the true carb amount. A high
# sigma is the dominant source of meal-bolus crashes that push TBR1 well
# above the real-cohort band, so this is kept modest.
CARB_COUNT_ERROR_SIGMA_BASE = 0.60  # [HIVAR] 0.30→0.60 — wider carb-count error (drives more post-meal excursions both ways).

# Asymmetric carb-count bias. Real T1D patients err on the side of under-bolusing
# because they fear hypos more than mild post-meal hypers (the classic "round
# down" rule). Without this bias, symmetric N(0, σ) carb-count errors crash as
# many patients as they spare; with it, the distribution shifts so the typical
# meal bolus is ~8% smaller than carb-count would imply, moving population time
# from TBR1 into TAR1 (180-250) — the band most under-represented vs real data.
CARB_COUNT_UNDERBOLUS_BIAS = 0.0    # Zeroed. Stepped -0.20 → -0.10 → 0.0. With the shorter
                                     # the larger value, every dinner was systematically
                                     # under-bolused by 20%, leaving residual carb load that
                                     # drove the +50 mg/dL evening overshoot vs Ohio. Educated
                                     # T1Ds in well-controlled cohorts dose closer to truth.

# Insulin stacking
CGM_CHECK_INTERVAL_ATTENTIVE = 20  # Minutes between checks for attentive patient
CGM_CHECK_INTERVAL_INATTENTIVE = 240  # Minutes for inattentive patient
PATIENCE_TIME_COMPETENT = 120  # Minutes before re-correcting (competent). IOB-aware correction
                               # sizing keeps rebound bounded; real T1D pump users typically correct
                               # every ~2 h when high. Longer patience drops correction frequency
                               # well below the real-cohort rate.
PATIENCE_TIME_INCOMPETENT = 60  # Minutes before re-correcting (incompetent)
CORRECTION_FACTOR_MEAN = 40.0  # mg/dL drop per unit of insulin
CORRECTION_FACTOR_SIGMA = 20.0  # [HIVAR 2x] 10.0→20.0 — correction-factor scatter
BG_TARGET = 158.0  # Target BG for corrections. Aims near the real-cohort median (155–157) — corrections targeting lower routinely overshoot into hypo because absorbed insulin keeps acting after BG arrives at target. Sits well above the ATTD ideal (~110) — sim correction
                   # kinetics + delivery lag tend to overshoot, so a higher target keeps median BG
                   # near the real OhioT1DM ~157 and TBR1 near real's ~3% rather than runaway hypo.
BG_HIGH_THRESHOLD = 175.0  # Threshold to trigger correction. Paired with `BG_TARGET=158`, corrections fire moderately above target.
BG_LOW_THRESHOLD = 60.0  # Floor of the hypo band; the acting trigger is the per-patient PatientProfile.hypo_threshold below.
# [CF] Per-patient "what I treat as low", centred on HYPO_THRESHOLD_MEDIAN and
# spread across the skill range: attentive, competent patients act earlier (higher
# BG). This one value drives the rescue AND blocks every bolus beneath it.
HYPO_THRESHOLD_MEDIAN = 80.0      # threshold of a median-skill patient (mg/dL)
HYPO_THRESHOLD_SKILL_SPAN = 10.0  # mg/dL deviation at each end of the skill range, so the population spans MEDIAN +/- this
HYPO_THRESHOLD_SKILL_MID = 0.5    # population-median skill_avg; the gain is applied piecewise about it because SKILL_MIN/MAX are NOT symmetric around it (0.15 vs 0.98), so one linear gain cannot hit +/-SPAN at both ends and hold the median

# Pre-meal bolus BG-awareness. Real T1Ds glance at their CGM before injecting
# the meal bolus; if they're already low or trending toward low, they skip,
# delay, or reduce the dose. Without this gate a scheduled meal pre-bolus
# pumps insulin into an actively hypoglycemic patient (the dominant sawtooth
# driver — patient eats correction carbs, gets briefly above 70, then the
# pre-scheduled meal bolus drags them straight back down).
# [CF] The skip threshold is now PER PATIENT — PatientProfile.hypo_threshold, the
# same value that triggers this patient's rescue. A global 75 mg/dL made a
# cautious patient who treats at 78 still take their meal bolus, and a stoic one
# who treats at 63 skip a dose they would in fact have given. One definition of
# "low" per patient, governing the rescue and every bolus alike.
BOLUS_REDUCE_MARGIN = 30.0         # Band above hypo_threshold in which the bolus is reduced rather than skipped (was the gap between the old global 75 and 105).
BOLUS_REDUCE_FACTOR_BASE = 0.3     # Reduction floor — multiplied by (1 + 0.3*dosing_competence). When the BG-aware gate engages near borderline-low, the bolus is halved (or more) rather than scaled down marginally.
BOLUS_BG_CHECK_BASE_PROB = 0.95    # Probability a patient checks CGM before bolusing; +0.05*attentiveness. Essentially every bolus is gated.
MEAL_BOLUS_SKIP_BASE_PROB = 0.08   # 0.25->0.08 — the high skip rate was a pre-OU ACF trick (un-bolused meals as decorrelating shocks). The Ornstein-Uhlenbeck glucose-effectiveness equilibrium now carries the ACF, so the heavy skipping is vestigial: restoring most meal bolusing lifts the (too-low) bolus share and sharpens the diurnal amplitude toward Ohio WITHOUT raising the 8h ACF (verified: acf8h ~0.02 either way). Still weighted by (1 - s3) for realistic occasional misses.
AFTERNOON_SNACK_SLOT = 2            # [GE-FIX] index into MEAL_TIME_OFFSETS_HOURS of the mid-afternoon eating occasion
AFTERNOON_SNACK_BOLUS_SKIP_BASE = 0.15  # 0.55->0.15 — likewise vestigial now the OU carries the ACF/afternoon shape; a modest residual keeps some real grazing un-bolused. Weighted by (1 - 0.5*s3).

# Hypo correction
HYPO_CORRECTION_BASE_GRAMS = 8.0  # Base correction (still under rule-of-15 of 15g). Kept at 8 — raising further makes the severe-hypo deficit-scaling formula (`14 + 0.35*deficit`) no longer dominate, breaking the test that deeper hypo yields strictly more rescue carbs.
HYPO_PANIC_FACTOR_BASE = 1.0  # How much extra is eaten, scaled by 1/s3
HYPO_DETECTION_AWAKE_MINUTES = 5.0  # Detection delay awake
HYPO_DETECTION_ASLEEP_LAMBDA = 30.0  # Exponential mean for detection delay asleep (severe hypo bypasses this)

# Exercise
EXERCISE_PROBABILITY_BASE = 0.3  # Base daily probability
EXERCISE_SKILL_BONUS = 0.4  # Added probability from s4
EXERCISE_TIME_MEAN_OFFSET_HOURS = 11.0  # Typical time: wake + 11h. For the 7:30 wake population this lands the
                                        # average session around 18:30 (after-work / pre-dinner). Earlier values
                                        # (9.0 → ~16:30) clustered with the old 16:30 dinner; with dinner now at
                                        # 19:30 the exercise offset is shifted in step so the two events stay
                                        # ordered (exercise → dinner) rather than overlapping.
EXERCISE_TIME_SIGMA_HOURS = 4.0  # [HIVAR] 2→4 — wider exercise-timing scatter.
EXERCISE_DURATION_MEAN_MIN = 75.0       # Population-mean session length. Real OhioT1DM mean ≈ 86 min,
                                        # σ across patients ≈ 57; the previous 40-min mean represented
                                        # only short walkers. Each patient now samples their own mean
                                        # from N(75, 45) so the population spans walkers to cyclists.
EXERCISE_DURATION_MEAN_SIGMA_MIN = 90.0  # [HIVAR] 45→90 — wider across-patient session-length spread.
EXERCISE_DURATION_MEAN_MIN_CLAMP = (12.0, 225.0)  # [HIVAR 2x] (15.0, 200.0)→(12.0, 225.0) — session-length clamp
EXERCISE_DURATION_SIGMA_MIN = 40.0  # [HIVAR 2x] 20.0→40.0 — within-patient session-length noise
EXERCISE_CARB_EQUIV_PER_MIN = 0.5  # Negative carb equivalent per minute of exercise
# [CF] A session is planned hours ahead but only starts if BG allows it. Blocked
# below hypo_threshold + this margin: exercise is negative food, so starting a
# workout already low drives BG straight down, and nobody sets off on a run at a
# BG they would treat at rest. The margin sits above the rest-state threshold
# because the bar for starting exercise is higher than the bar for eating —
# clinical guidance is to top up with carbs below ~90 mg/dL before activity.
EXERCISE_HYPO_MARGIN = 20.0
EXERCISE_GAMMA_K = 3.0
EXERCISE_GAMMA_THETA = 15.0

# Hepatic glucose output (insulin-suppressed via Hill function)
# At zero insulin, HGO runs near UNSUPPRESSED. As plasma insulin rises, HGO
# saturates toward the SUPPRESSED floor. HGO_INSULIN_HALF_MAX is tuned so that
# a typical basal level (~0.07 U/step) lands at the legacy ~9 g/hr rate, which
# preserves the basal-balances-HGO test invariant.
HGO_BASE_GRAMS_PER_HOUR = 8.25  # "Balanced" reference rate used for basal sizing
                                 # (ideal_basal = HGO_BASE * 24 * weight/75 * is / ICR).
                                 # Lowered from 9.0 → 8.25 jointly with the ICR_MEAN 11→8
                                 # drop: at HGO=9 + ICR=8 the simulator's basal share ran
                                 # 58% (mean rate 1.32 U/hr) while AZT1D's pump log shows
                                 # ~0.92 U/hr at ~43% basal share. 8.25/8 = 24.75 U/day
                                 # basal lands the mean rate at ~0.94 U/hr (matches AZT1D
                                 # within rounding) and brings basal share to ~53%. Going
                                 # lower (e.g. 7.5) overshoots the basal-share target but
                                 # collapses mean BG dynamics because the steeper Hill
                                 # curve required for anchoring robs meal boluses of their
                                 # extra dynamic HGO-suppression power — post-meal BG
                                 # climbs +15 mg/dL.
HGO_UNSUPPRESSED_GRAMS_PER_HOUR = 18.0  # Rate with no insulin (DKA-like)
HGO_SUPPRESSED_FLOOR_GRAMS_PER_HOUR = 6.0  # Maximum suppression
HGO_INSULIN_HALF_MAX = 0.0198  # Insulin per step at which HGO is half-suppressed (U/step).
                                # Retuned 0.025 → 0.0198 jointly with HGO_BASE so that
                                # Hill(typical_basal_per_step) ≈ HGO_BASE under the new
                                # ICR/HGO calibration. With ICR=8 and HGO_BASE=8.25,
                                # typical basal-per-step is 8.25*24/8/288 ≈ 0.0859 U/step;
                                # the closed-form solution for the anchor constraint
                                # `Hill(B/96) = B` is h = B*(B-6) / (96*(18-B)) =
                                # 8.25*2.25/(96*9.75) ≈ 0.0198. Mild change from 0.025 so
                                # the Hill curve remains close to its previous shape and
                                # meal-bolus HGO suppression dynamics are preserved.
HGO_NOISE_SIGMA = 0.0  # [MINNOISE] step-to-step hepatic-output noise minimized to zero
HGO_INSULIN_SMOOTHING_ALPHA = 0.25  # EMA factor for the insulin level fed into the Hill function.
# Models plasma-insulin lag behind SC absorption (~10-15 min), and prevents HGO
# from stepping when a new bolus curve activates. Half-life ≈ 12 min at α=0.25.

# Circadian HGO modulation — cortisol drives a dawn surge in hepatic glucose
# output (peaks ~6-7am), and HGO dips during deep sleep (~2-3am). Without this,
# the simulator misses the canonical dawn phenomenon and instead shows a
# nighttime BG rise driven only by basal ramp-down + delayed-meal HGO.
# Multipliers are applied to hgo_value after the Hill computation, so they
# stack with insulin suppression: a well-bolused patient still sees a smaller
# dawn rise. Per-patient amplitude is sampled in generate_patient (see
# patient.dawn_hgo_amplitude / patient.night_hgo_dip_amplitude) so individuals
# can have stronger or weaker dawn effects.
DAWN_HGO_PEAK_HOUR = 8.0             # [DIURNAL] Hour of peak dawn HGO surge. 6.0 -> 8.0: at 6.0 the
                                     # fast HGO surge dominated the morning rise and put the sim's
                                     # diurnal maximum at hour 6-7, an hour early vs Ohio's hour-8 peak.
                                     # 8.0 aligns the surge with waking and the GE dawn profile (also
                                     # h8), landing the morning BG peak on hour 8 as in the real cohorts.
DAWN_HGO_SIGMA_HOURS = 2.5           # Gaussian width. Narrowed back from 3.5 → 2.5 so the surge
                                     # spans roughly midnight-to-noon rather than 5am-10am.
                                     # Wider sigma raises the daily HGO integral, intentionally
                                     # — the previous narrow surge couldn't sustain the
                                     # midnight-to-6am overnight rise that Ohio and Shanghai
                                     # both show. Per-patient amplitude noise is unchanged
                                     # so individual variability is preserved.
DAWN_HGO_AMPLITUDE_MEAN = 6.0        # [GE-FIX] 8.5→6.0 — trimmed so the morning BG peak (~186 like Ohio, was ~193) doesn't overshoot and inflate the daily amplitude. Mean peak HGO surge in g/hr (additive). 9 → 11 → 8.5:
                                     # to drive the overnight BG rise that real cohorts show
                                     # (Ohio: +24 mg/dL from midnight to 6am; sim used to be
                                     # −33 over the same window). Combined with the wider
                                     # sigma and earlier peak, the daily HGO integral grows
                                     # from ~40 to ~96 g/day. Compensated by smaller meals
                                     # (MEAL_CARB_MEANS trimmed) so net daily glucose budget
                                     # is roughly preserved.
DAWN_HGO_AMPLITUDE_SIGMA = 4.0  # [HIVAR 2x] 2.0→4.0 — dawn-surge spread
NIGHT_HGO_DIP_HOUR = 2.0             # Hour of deep-sleep HGO trough
NIGHT_HGO_DIP_SIGMA_HOURS = 2.5      # Narrower so it ends before dawn surge starts
NIGHT_HGO_DIP_AMPLITUDE_MEAN = 3.5   # [DIURNAL] 0.3 -> 3.5. Mean deep-sleep HGO reduction in g/hr. A
                                     # strong overnight HGO trough (deep sleep + fasting genuinely
                                     # suppress hepatic output) is what pulls the overnight window BELOW
                                     # the daytime, giving Ohio's real day>night diurnal — the small 0.3
                                     # left the sim inverted (night>day) under the delayed-HGO dinner
                                     # rebound. The larger dip only mildly raises TBR (2.6->2.8) here
                                     # because the GE floor + basal feedback keep the overnight lows bounded.
NIGHT_HGO_DIP_AMPLITUDE_SIGMA = 0.5  # [HIVAR 2x] 0.25→0.5 — night-dip spread
# Daily-integrated contribution (Gaussian: A * sigma * √(2π)):
#   dawn ≈ 9.0 * 1.8 * √(2π) ≈ 40.6 g/day extra
#   dip  ≈ 0.7 * 2.5 * √(2π) ≈ 4.4  g/day reduction
# Net ≈ +36 g/day glucose-in (intentional — without a strong dawn surge the
# canonical dawn phenomenon doesn't appear at all; the breakfast bolus
# cancels most of the morning rise).

# Glycogen reservoir — finite hepatic glycogen store that drains under HGO and
# refills from absorbed carbs. When depleted the liver loses its glycogenolysis
# source and HGO scales down toward a gluconeogenesis-only floor. Without this
# the unsuppressed-HGO state would be an infinite battery (no fasting limit).
GLYCOGEN_CAPACITY_GRAMS = 100.0  # Maximum hepatic glycogen
GLYCOGEN_INITIAL_FRACTION = 0.7  # Patients start moderately full
GLYCOGEN_DRAIN_FRACTION = 0.35  # Fraction of HGO sourced from glycogenolysis (rest is gluconeogenesis).
                                 # Lowered from 0.5: at 0.5, overnight glycogen depletion halved
                                 # the dawn HGO surge exactly when it was most needed (hour
                                 # 03-05), undoing the timing-shift gains. Real T1Ds keep
                                 # gluconeogenesis active overnight (Cahill cycle from amino
                                 # acids + glycerol), so a 65/35 gluconeogenesis/glycogenolysis
                                 # split is more physiologically accurate than 50/50.
GLYCOGEN_REFILL_FRACTION = 0.20  # Fraction of absorbed carbs stored as glycogen
GLYCOGEN_LOW_THRESHOLD_FRACTION = 0.15  # Below this fraction of capacity, HGO ramps down

# Glucotoxicity — sustained hyperglycemia transiently increases insulin
# resistance ("glucose toxicity"). Slow EMA of BG drives an additive IS factor.
# Closes a positive feedback loop: high BG → more IR → harder to bring down.
GLUCOTOX_BG_EMA_HALF_LIFE_HOURS = 3.0  # Short half-life — longer memory contributes to a 4–12h ACF plateau that real CGM doesn't show.
GLUCOTOX_BG_THRESHOLD = 200.0  # Above this EMA value, IS starts to climb
GLUCOTOX_BG_FOR_MAX = 350.0  # EMA value at which the maximum IR multiplier is applied
GLUCOTOX_MAX_IS_INCREASE = 0.15  # Up to 15% more resistant at saturating BG

# Postprandial insulin resistance — in non-diabetics the incretin / GLP-1 axis
# augments insulin secretion and sensitivity around a meal. In T1DM that axis is
# blunted and there is no endogenous insulin response, so the meal-time
# sensitivity boost is absent; if anything the absorbing-carb state is mildly
# insulin-RESISTANT (glucose/FFA surge, absent amylin → faster gastric emptying,
# counter-regulatory tone). While carbs are absorbing, insulin clears glucose
# slightly LESS effectively. Saturating in active carb, peaks at a ~4% penalty.
POSTPRANDIAL_IR_PENALTY_FACTOR = 0.04
POSTPRANDIAL_IR_PENALTY_HALF = 1.5  # g/step active carb at half-max penalty

# Injection site quality (lipohypertrophy) — per-dose multiplier on the
# delivered insulin. Sigma scales inversely with lifestyle_consistency (poor
# rotation discipline → more variance and occasional poor sites).
SITE_QUALITY_SIGMA_BASE = 0.2  # [HIVAR 2x] 0.1→0.2 — per-dose absorption scatter
SITE_QUALITY_MIN = 0.35  # [HIVAR 2x] 0.5→0.35 — malabsorption floor
SITE_QUALITY_MAX = 1.65  # [HIVAR 2x] 1.4→1.65 — absorption-surge ceiling

# Delayed-meal HGO rebound — large meals trigger a positive HGO bump 3.5-5.5h
# later (delayed gluconeogenesis from amino acids + cortisol response). This
# is the mechanism behind nocturnal hyperglycemia after a big dinner.
DELAYED_HGO_MEAL_THRESHOLD_GRAMS = 60.0  # Meals above this trigger a rebound
DELAYED_HGO_PER_GRAM = 0.015  # [DIURNAL] 0.02->0.015 g/hr of HGO bump per gram of meal carbs above threshold — the dinner rebound (nocturnal hyperglycemia) is retained but trimmed so the strengthened night HGO dip can land the overnight window below the day
DELAYED_HGO_MAX_BUMP = 3.5  # [DIURNAL] 5.0->3.5 Cap on HGO bump magnitude (g/hr)
DELAYED_HGO_DELAY_HOURS_MIN = 3.5  # Earliest onset after meal
DELAYED_HGO_DELAY_HOURS_MAX = 5.5  # Latest onset
DELAYED_HGO_DURATION_HOURS_MIN = 2.7  # [HIVAR 2x] 3.0→2.7 — rebound-duration span
DELAYED_HGO_DURATION_HOURS_MAX = 7.0  # [HIVAR 2x] 6.0→7.0 — rebound-duration span
DELAYED_HGO_RAMP_HOURS = 1.0  # Trapezoidal ramp up/down for the rebound envelope

# Per-step absorption noise on the carb/insulin reads. Models gut absorption
# variability (mixing, blood flow) and subcutaneous depot dissolution variance.
# Multiplicative — only matters when the underlying curve is non-zero.
CARB_ABSORPTION_NOISE_SIGMA = 0.0  # [MINNOISE] step-to-step carb-absorption noise minimized to zero
INSULIN_ABSORPTION_NOISE_SIGMA = 0.0  # [MINNOISE] per-step insulin-absorption noise removed — basal/bolus curves were sawtoothing

# BG computation
BG_SCALE_FACTOR = 3.5  # [GE-FIX] 1.5→3.5 — restored toward the pre-DAMP scale now that glucose effectiveness (below) supplies the in-band restoring force. The prior 1.5 damped every flux term including the patient's own correction insulin, slowing the correction loop and lengthening the BG memory (long-lag ACF). With Sg bounding excursions, alpha can carry realistic 5-min delta scale and faster correction dynamics again.
BG_CLAMP_MIN = 10.0  # [CF] 40→10 — the 40 floor was the real CGM device floor, but a device that stops REPORTING at 40 does not stop the patient falling. Pinning the dynamics there taught every downstream model that a descent tapers out near 40 and that a dangerous low is unreachable. 10 mg/dL is below survivable, so it never binds physiologically; it exists only to keep the Kovatchev log transform defined (f is -inf at 0, NaN below 1). Risk space is now asymmetric, f(10) = -6.82 vs f(400) = +3.16, which correctly prices a deep low above a high.
BG_CLAMP_MAX = 400.0  # [HIVAR] 500→400 — match real CGM device ceiling (Ohio/AZT1D max = 400).
BG_INITIAL_MEAN = 120.0
BG_INITIAL_SIGMA = 60.0  # [HIVAR] 30→60 — wider warmup start (washes out over the run).
BG_INITIAL_FLOOR = 40.0  # [CF] a run starts at a plausible BG, not at the dynamics clamp. Clipping the N(mean, 60) draw with BG_CLAMP_MIN would start ~4% of patients pinned in severe hypo before a single step ran; the warmup washes it out, but the transient is longer now that Sg is weak.

# Soft BG bounds: in the approach zone, a single step can close at most
# SOFT_APPROACH_FRACTION of the remaining headroom to the hard bound. This
# gives geometric asymptotic decay toward the floor/ceiling — BG never
# actually reaches the hard clamp under normal dynamics, regardless of how
# large the raw delta is. The hard clamp is kept only as a backstop.
BG_SOFT_FLOOR = 20.0           # [CF] 50→20 — tracks BG_CLAMP_MIN down. Left at 50 the damping engaged 30 mg/dL above the new floor and re-imposed the taper the floor change removes.
BG_SOFT_CEILING = 385.0        # [HIVAR] 400→385 — keep a soft runway below the new 400 hard ceiling
SOFT_APPROACH_FRACTION = 0.15  # [DAMP] 0.3→0.15 — stronger delta-damping as BG nears the bounds.

# BG regulatory computation
RENAL_THRESHOLD = 180.0  # Kidneys start excreting glucose above this
RENAL_CLEARANCE_RATE = 0.015  # [DAMP] 0.005→0.015 — arrest hyper excursions faster (shorter/shallower highs).
COUNTER_REGULATORY_THRESHOLD = 70.0  # Body releases glucagon below this
COUNTER_REGULATORY_RATE = 1.5  # [DAMP] 0.8→1.5 — arrest hypo excursions faster.
SEVERE_HYPO_THRESHOLD = 55.0  # Below this, glucagon dump kicks in
SEVERE_HYPO_GLUCAGON_RATE = 2.0  # Extra mg/dL per step at severity=1.0

# Glucose effectiveness (Bergman minimal-model Sg): the insulin-INDEPENDENT
# glucose disposal + hepatic-glucose-output autoregulation that clears a
# glucose load even at basal insulin. Modeled as an always-on linear restoring
# pull of the BG delta toward a STOCHASTIC equilibrium E(t):
#     bg_delta += glucose_effectiveness * (E(t) - bg)
# E(t) is an Ornstein-Uhlenbeck process: each step it mean-reverts (timescale
# GE_EQ_TAU_HOURS) toward the patient's anchor plus a dawn-phenomenon lift
# (ge_anchor + ge_dawn_amplitude * ge_diurnal_profile(hour)), perturbed by Gaussian
# noise of stationary std GE_EQ_SIGMA * ge_sigma_mult, then floored at GE_EQ_FLOOR so the
# restoring target is never *severely* hypoglycemic (the floor sits above the
# 55 mg/dL severe threshold). Sg is the mean-reverting force the
# renal/counter-regulatory guardrails do NOT supply inside the normal 70-180
# band (they are gated to BG>180 / BG<70). Without it, within-band BG is an
# under-damped integrator of net flux — a slowly-corrected random walk — so the
# level autocorrelation decays far too slowly (acf at 8h ~0.3 vs ~0 in real CGM).
# The Sg pull keeps BG's correlation time short (low 8h ACF); E's
# wandering supplies distributional spread but decorrelates within hours, so it
# does NOT re-inject long-lag memory — decoupling spread from the ACF, which a
# fixed setpoint could not do (a fixed target damped the ACF only by homogenizing
# the distribution). E's own timescale carries most of that decorrelation, which
# is why Sg can be weak without the 8h ACF returning; keep it weak, because the
# spring cancels the sustained component of any input slower than its own time
# constant, insulin included. GE_RATE is the population-mean per-step reversion fraction;
# per-patient Sg is sampled lognormally around it (real Sg varies ~2-3x across
# individuals) so heterogeneity is preserved. The GE_EQ_FLOOR is what keeps the
# Sg pull *upward* in a low (aiding, never opposing, the severe-hypo rescue) and
# simultaneously anchors the pooled mean/low-tail onto the real cohorts. See
# ge_day_weight for the diurnal lift and the OU update in generate().
GE_RATE = 0.015            # [CF] 0.060→0.015 — the spring was installed to pull acf(8h) down, but E's own 3h OU timescale carries that: two independent measurements (24 x 30 d and 64 x 30 d) put acf(8h) at +0.03..0.05 before and +0.05..0.06 after, so the spring was never what held it near zero — E's own timescale is. What the spring cost was causality. Time constant 1/0.060 = 83 min high-passes any input slower than itself; insulin acts over 3-5 h, so against a Sg=0 reference of -31.1 mg/dL a 2 U bolus retained only ~20% of its 240-min effect (-6.3). At 0.015 it retains ~87% (-27.0). Price paid, same protocol: acf(2h) 0.49 -> 0.55 — note it was ALREADY on Ohio's 0.483 and moves past it toward Shanghai's 0.606, so the ACF is a cost here, not a justification — acf(24h) 0.09 -> 0.16, and time<55 0.65% -> 1.53% mean / 2.42% -> 4.05% worst patient (invariant: mean<=2%, max<=6%).
GE_EQ_ANCHOR_MEAN = 138.0  # [GE-OU] population-mean equilibrium anchor (co-tuned with GE_EQ_SIGMA / GE_EQ_FLOOR to land the pooled mean ~162 on Ohio; with a large sigma the floor-clipping largely sets the mean, so the anchor is a fine trim)
GE_EQ_ANCHOR_SIGMA = 15.0  # [GE-OU][HET] between-patient spread of the anchor (per-patient mean heterogeneity). Raised 12->15 to widen the too-narrow between-patient mean-BG spread (sim was the narrowest of the four cohorts, ~13.6 vs Ohio/AZT1D ~16). The anchor sigma alone SATURATES near ~14 mg/dL sd because GE_EQ_FLOOR compresses low-anchor patients upward; the remaining spread comes from the IR coupling below (high-side, floor-immune).
GE_ANCHOR_IR_COUPLING = 30.0  # [GE-OU][HET] how much a patient's insulin resistance lifts its equilibrium anchor (mg/dL per unit ir-1). Raised 12->30 (with IR_LOGNORMAL_SIGMA 0.16->0.26) so the between-patient mean spread is carried physiologically by the resistance axis — resistant patients run higher — rather than by low-anchor patients the floor would both compress and crash into hypo.
GE_EQ_SIGMA = 95.0                # [GE-OU][OHIO-CARB][DAWN] stationary std of the equilibrium's APERIODIC wandering. Was 105; lowered to 95 when the structured dawn rhythm (GE_EQ_DAWN_AMPLITUDE) took over part of the equilibrium's variance — the point of the dawn work is to trade opaque aperiodic wander for day-to-day-consistent, learnable variance while holding the pooled BG spread (std ~61) on Ohio. Now scaled PER PATIENT by ge_sigma_mult (below) so patients differ in within-patient variability.
# [HET] Per-patient multiplier on GE_EQ_SIGMA (lognormal), so different virtual
# patients wander with different amplitudes. Without it every patient shared the
# one global GE_EQ_SIGMA, making the between-patient spread of WITHIN-patient
# variability (std/CV/MAGE) the narrowest of the four cohorts. E[mult^2]>1 lifts
# the mean within-variance, so GE_EQ_SIGMA base is co-trimmed to hold pooled std.
GE_SIGMA_REL_SIGMA = 0.16  # lognormal sigma of the per-patient GE_EQ_SIGMA multiplier. Tamed 0.25->0.22->0.16: the wider IR axis (for the mean spread) already stretches insulin sensitivity, so a large wander multiplier on top re-deepens rare over-bolus crashes; at 0.16 the between-patient within-variability spread still lands in the real range (wstd sd ~7.8, Ohio 7.2) while the 300x70d severe-hypo tail is no worse than baseline (>2h=0 vs baseline 1).
GE_SIGMA_MULT_CLIP = (0.68, 1.38)  # clip the multiplier's tails
GE_EQ_TAU_HOURS = 3.0      # [GE-OU][TEXTURE] OU timescale. 2.0 -> 3.0: a slightly longer correlation time smooths the equilibrium's step-to-step wander, cutting the excess signal complexity / hyper-episode fragmentation (SampEn ~1.07 -> ~0.9, hyper/day 3.0 -> 2.8 toward the real cohorts) AND raising the mid-lag autocorrelation (2h/4h) toward Ohio, which the too-short 2h timescale left below the real values. Still << 8h, so the 8h ACF stays ~0 (no long-lag memory).
GE_EQ_DAY_BOOST = 10.0     # [GE-OU] legacy flat daytime lift — SUPERSEDED by the dawn-phenomenon profile below (kept as a constant for backward-compat); the OU step now uses ge_diurnal_profile instead.
# [DAWN] Structured dawn-phenomenon rhythm on the OU equilibrium. Replaces the flat
# day boost with a per-patient daily profile (peaks post-dawn), tied to each patient's
# dawn trait. This converts opaque OU wander into a signal that is a deterministic
# function of time-of-day AND a per-patient trait — hence LEARNABLE by a downstream
# behaviour->BG model (a real dawn rise it can predict), which pure OU noise is not.
# Co-tuned DOWN against GE_EQ_SIGMA so total spread stays on Ohio while the day-to-day-
# consistent (periodic) share of the variance rises -> higher acf24h and diurnal amplitude.
GE_EQ_DAWN_AMPLITUDE_MEAN = 35.0   # [DAWN] population-mean amplitude of the equilibrium's daily rhythm (mg/dL); per-patient value scales with the dawn trait
GE_DAWN_PEAK_HOUR = 8.0            # [DAWN] hour of the equilibrium's daily peak (Ohio's diurnal peak is h8 / post-waking cortisol rise)
GE_DAWN_WIDTH_HOURS = 5.5          # [DAWN] Gaussian width of the daily profile — broad, so daytime stays elevated and only the small hours dip
GE_DAY_START_HOUR = 7.0    # setpoint ramps up around wake
GE_DAY_END_HOUR = 22.0     # ramps back down in the late evening
GE_DAY_RAMP_HOURS = 3.0    # smootherstep ramp width for the day/night setpoint transitions
GE_REL_SIGMA = 0.30        # per-patient lognormal spread of Sg around GE_RATE (~2x inter-individual range)
GE_RATE_MIN = 0.004        # floor so no patient is a pure (undamped) integrator
GE_RATE_MAX = 0.150        # [GE-OU] raised so the strong per-patient Sg (lognormal around GE_RATE) is not clipped
GE_EQ_FLOOR = 64.0         # [GE-OU][OHIO-CARB][DIURNAL] floor on the wandering equilibrium, kept above the severe-hypo threshold (SEVERE_HYPO_THRESHOLD=55) so the Sg pull stays upward across a severe low (aids, never opposes, the rescue). Journey 75->60 (OU/carb co-tune) then 60->64 (diurnal fix): the strengthened deep-sleep HGO dip that gives the real day>night shape also removes overnight glucose, so a slightly higher floor lifts the low tail back onto Ohio's mean AND strengthens the overnight rescue pull (TBR 3.3->3.1), while severe episodes stay bounded (>2h = 0).

# CGM noise
# CGM interstitial lag. The sensor sits in interstitial fluid, which trails
# plasma glucose by a first-order diffusion process (Rebrin/Steil):
#   dIG/dt = (BG - IG) / tau,  tau = CGM_LAG_MINUTES.
# Applied in `_compute_cgm_observation` before the sensor noise, so every
# consumer of bg_observed (corrections, hypo detection, the exported CGM
# channel) sees the delayed-and-smoothed interstitial value, not instant BG.
CGM_LAG_MINUTES = 15
CGM_NOISE_FRACTION = 0.060  # [GE-FIX] 0.120→0.060 — reverted the HIVAR 2x back to the value calibrated for the ~5.8 mg/dL Δ5min-std target; the true-BG delta scale is now carried by BG_SCALE_FACTOR, and observed acf(8h) stays ~true (~0.03) since sensor noise no longer dilutes it.
                            # (~9 mg/dL drift around BG=150). Bumped from 0.018 to compensate
                            # for the AR(1) step-to-step variance reduction; preserves the
                            # 5.81 mg/dL Δ5min std target while producing the smooth,
                            # correlated "Perlin-like" wobble seen in real CGM data instead
                            # of the white-noise spikiness independent draws produced.

# AR(1) correlation for noise sources. Replaces independent per-step Gaussian
# draws with Ornstein-Uhlenbeck-like smooth variability. ρ=0.85 for metabolic
# noises (~22 min correlation half-life); ρ=0.92 for the CGM sensor (~42 min,
# matches documented Dexcom/Libre ARMA models).
NOISE_AR1_RHO_METABOLIC = 0.85
NOISE_AR1_RHO_SENSOR = 0.92
# Pre-computed √(1 − ρ²) — the per-step innovation scale that preserves
# stationary variance σ² when iterating x_t = ρ·x_{t−1} + scale·ε_t.
NOISE_AR1_INNOV_METABOLIC = float(np.sqrt(1.0 - NOISE_AR1_RHO_METABOLIC ** 2))
NOISE_AR1_INNOV_SENSOR = float(np.sqrt(1.0 - NOISE_AR1_RHO_SENSOR ** 2))

# Rare events
RARE_EVENT_PROBABILITY = 0.04  # [HIVAR 2x] 0.02→0.04 — chaotic-day rate
RARE_EVENT_SKILL_REDUCTION = 0.3  # Even skilled people have bad days sometimes

# Hypo correction refractory + post-hypo basal stand-down. Without the basal
# stand-down a hypo cascades into 3-5 consecutive corrections (or a sawtooth
# of snacking) because the forward basal pipeline keeps clearing glucose just
# as the patient eats to recover. Real patients respond by reducing or
# suspending basal coverage — for pump users a temp basal / pump suspend,
# for MDI users skipping the next basal injection — not by stacking carbs.
HYPO_CORRECTION_REFRACTORY_MIN = 20.0  # Min minutes between hypo corrections (moderate hypo 55-70).
SEVERE_HYPO_REFRACTORY_MIN = 10.0      # Shorter refractory for severe hypo (<55). Rule-of-15 spirit:
                                       # symptomatic patient still rage-eats, but waits ~10 min between
                                       # doses so the first rescue's carbs can act. Without this gap,
                                       # the CGM-check bypass let the patient eat every 5 min, stacking
                                       # 3-5 rage doses (60+ g) and producing visible sawtooth as BG
                                       # bounced between severe hypo and post-overcorrection peaks.
POST_HYPO_BASAL_SUSPEND_DURATION_HOURS = 6.0  # Scale-down window after any hypo correction. The sin² envelope below has max slope π/window, so a 6h window with depth 0.35 caps the worst single-envelope 10-min change at ~3%, and even at multi-envelope crossover points the per-10-min change stays under ~6%. Widened from the legacy 2h trapezoidal window after the user observed ~56% 10-min drops in total_insulin.
POST_HYPO_BASAL_SUSPEND_FACTOR = 0.65          # Basal contribution multiplier at the peak of the suspend window. Shallower than the legacy 0.35 — the integrated suspend stays close to the original (0.35 × 0.5 × 360 ≈ 63 unit-minutes vs legacy ~65) but spread over a wider window so per-step changes are gentle.
POST_HYPO_BASAL_SUSPEND_RAMP_MIN = 20.0        # Legacy constant from the previous smootherstep-plus-plateau implementation. The active envelope is now a single sin² profile over the full duration window (no separate ramp/plateau), so this constant is unused — kept only for backward compatibility with anything that imports it.

# Rage behavior
RAGE_EAT_BG_THRESHOLD = 50.0       # Below this, patient may rage eat
RAGE_EAT_CARB_MIN = 12.0           # Minimum rage eat carbs
RAGE_EAT_CARB_MAX = 30.0           # Maximum rage eat carbs
RAGE_EAT_PROBABILITY_BASE = 0.10   # Base chance of rage eating when below threshold
# [CF] Rule-of-15's second half: "wait, RECHECK, and only re-dose if still low".
# The refractory timer supplies the waiting; this supplies the recheck. Without
# it the patient re-treats purely on current BG, blind to the glucose already
# absorbing, and a deep low turns into a cascade of doses — 16 rescues and 253 g
# in one episode at the worst. Mirrors the IOB-aware insulin correction: a
# baseline everyone manages plus a competence-scaled bonus. The baseline is lower
# than insulin's 0.7 because a symptomatic hypo is a worse moment for arithmetic
# than dialling a correction, so even careful patients over-treat somewhat.
COB_AWARENESS_BASE = 0.40          # fraction of carbs-on-board even a careless patient accounts for
COB_AWARENESS_SKILL = 0.50         # additional fraction scaled by dosing_competence (max 0.90 total)
RAGE_BOLUS_BG_THRESHOLD = 300.0    # Above this, patient may rage bolus
RAGE_BOLUS_MULTIPLIER_MIN = 1.1    # Minimum dose multiplier during rage bolus.
RAGE_BOLUS_MULTIPLIER_MAX = 1.5    # Maximum dose multiplier during rage bolus. Capped low — wider
                                   # multipliers cause stacking-induced crashes.
RAGE_BOLUS_PROBABILITY_BASE = 0.05 # Base chance of rage bolusing when above threshold. Modest —
                                   # real patients above 300 usually take a measured correction
                                   # rather than rage-dose.

# ============================================================================
# WEEKDAY / WEEKEND PARAMETERS
# ============================================================================

SIMULATION_START_DAY_OF_WEEK = 0       # Starting day of week (0=Monday, 6=Sunday)
WEEKEND_WAKE_DELAY_HOURS_MIN = 1.0     # Min extra hours slept in on weekends/holidays
WEEKEND_WAKE_DELAY_HOURS_MAX = 2.0     # Max extra hours slept in on weekends/holidays
WEEKEND_MEAL_JITTER_MULTIPLIER = 1.5   # Meal timing variability multiplier on weekends
WEEKEND_CARB_INCREASE_FRACTION = 0.15  # Fraction by which carb amounts can increase on weekends
WEEKEND_EXERCISE_PROB_MULTIPLIER = 0.8 # Exercise probability multiplier on weekends

# Public holidays (non-weekend working days treated as weekend for behavior)
PUBLIC_HOLIDAYS_PER_YEAR_MIN = 10      # Minimum number of public holidays per year
PUBLIC_HOLIDAYS_PER_YEAR_MAX = 20      # Maximum number of public holidays per year

# ============================================================================
# EXERCISE: DELAYED INSULIN SENSITIVITY EFFECT
# ============================================================================

EXERCISE_IS_REDUCTION = 0.10           # IS reduction fraction post-exercise (10% more sensitive)
EXERCISE_IS_DURATION_HOURS = 6.0       # Duration of post-exercise IS boost (hours). Long enough to model the well-known evening-hypo pattern after morning exercise, but short enough not to contribute to a half-day BG ACF tail.
EXERCISE_IS_RAMP_HOURS = 1.0           # Trapezoidal ramp up/down for the IS boost envelope

# ============================================================================
# TREND-BASED ANTICIPATORY CORRECTIONS
# ============================================================================

TREND_CORRECTION_WINDOW_STEPS = 6      # BG history window for trend (6 steps = 30 min)
TREND_HIGH_RATE_THRESHOLD = 5.0        # mg/dL/step rising trend to trigger preemptive correction.
                                       # Modest threshold — flatter sustained climbs into the 200–250
                                       # zone need to trip the trend gate to keep hyper p90 / TAR2
                                       # bounded.
TREND_HIGH_BG_MIN = 145.0              # BG must exceed this for trend-based high correction.
                                       # `eff_high_thresh = 180 - 25*skill_avg` lands at 160 for
                                       # high-skill patients, so this threshold must sit below 160 or
                                       # the trend window collapses to empty for skilled patients. At
                                       # 145, the window is 145–160 for high-skill and 145–170 for
                                       # low-skill, allowing preemptive corrections on dinner climbs.
TREND_LOW_RATE_THRESHOLD = -5.0        # mg/dL/step falling trend to trigger preemptive carb
TREND_LOW_BG_MAX = 110.0               # BG must be below this for trend-based low correction. Attentive patients catch falling trends *well* before 70 — they eat preemptive carbs when BG is dropping through the 90–110 band.

# ============================================================================
# ALCOHOL MODELING
# ============================================================================

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

# ============================================================================
# STRESS AND HORMONAL EFFECTS
# ============================================================================

STRESS_PROBABILITY_BASE = 0.18         # Per-day base probability of a stress event
STRESS_LIFESTYLE_WEIGHT = 0.16         # How much lifestyle_consistency reduces stress prob
STRESS_IS_FACTOR_MIN = 1.2             # Minimum IS multiplier during stress (more resistant)
STRESS_IS_FACTOR_MAX = 1.5             # Maximum IS multiplier during stress. Capped low —
                                       # larger IR spikes for 2-6h are a dominant CV-widening factor
                                       # and trigger post-stress hypos when the spike subsides.
                                       # while bolus IOB was still active. 50% IR is still substantial.
STRESS_DURATION_HOURS_MIN = 2.0        # Minimum duration of elevated IS from stress (hours)
STRESS_DURATION_HOURS_MAX = 6.0        # Maximum duration of elevated IS from stress (hours)
STRESS_IS_RAMP_HOURS = 0.5             # Trapezoidal ramp up/down for stress envelope

# ============================================================================
# ANOMALOUS EVENTS
# ============================================================================

ANOMALOUS_EVENT_PROBABILITY = 0.01     # Per-day probability of an anomalous curve modification
ANOMALOUS_THETA_MULT_MIN = 1.5         # Min theta multiplier (slower absorption)
ANOMALOUS_THETA_MULT_MAX = 3.0         # Max theta multiplier (much slower absorption)
ANOMALOUS_K_MULT_MIN = 0.3             # Min k multiplier (flatter curve)
ANOMALOUS_K_MULT_MAX = 2.0             # Max k multiplier (sharper peak)

# ============================================================================
# DATA STRUCTURES
# ============================================================================

class CarbType(Enum):
    FAST = "fast"
    SLOW = "slow"
    PROTEIN_FAT = "protein_fat"


@dataclass
class PatientProfile:
    """A virtual patient's skill profile and derived parameters."""
    # Raw skills (0-1 after sigmoid)
    dietary_discipline: float = 0.5
    attentiveness: float = 0.5
    dosing_competence: float = 0.5
    lifestyle_consistency: float = 0.5

    # The BG this patient personally treats as low (see HYPO_THRESHOLD_MEDIAN).
    # Attentive, competent patients act earlier. This is the single definition of
    # "low" for this patient: it triggers the rescue AND blocks every bolus below
    # it — a patient who considers themselves low does not dose insulin, whatever
    # the meal plan said.
    hypo_threshold: float = HYPO_THRESHOLD_MEDIAN

    # Derived physiological parameters
    body_weight_kg: float = BODY_WEIGHT_MEAN_KG
    insulin_resistance_factor: float = 1.0  # >1 = resistant, <1 = sensitive
    is_base: float = 1.0
    icr: float = 10.0
    correction_factor: float = 40.0
    glucose_effectiveness: float = GE_RATE  # per-patient Bergman Sg (per-step reversion)
    ge_anchor: float = GE_EQ_ANCHOR_MEAN    # per-patient equilibrium anchor (OU reverts to this + diurnal lift)
    ge_sigma_mult: float = 1.0              # per-patient multiplier on GE_EQ_SIGMA (within-patient variability heterogeneity)
    basal_dose: float = 20.0

    # Insulin analogue assignment (one rapid bolus + one long-acting basal per
    # patient; see BOLUS_VARIANTS / BASAL_VARIANTS). basal_duration_hours is now
    # the PK *action* duration of the chosen basal analogue; the injection
    # cadence is basal_dose_interval_hours (once daily), decoupled from it.
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
    patience_time_min: float = 120.0
    carb_count_error_sigma: float = 0.15
    bolus_timing_mean: float = 0.0
    bolus_timing_sigma: float = 10.0
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
    last_correction_idx: int = -9999
    last_cgm_check_idx: int = 0
    day_number: int = 0
    # Persistent multi-day basal-dose drift. Each day the reactive
    # basal_adjustment (limited to ±22% on a 3-day rolling mean) leaks a
    # small fraction into this multiplier, so chronically over- or under-
    # dosed patients permanently shift their base dose over a few weeks
    # (matches real-world clinic-driven basal program updates). Initialized
    # to 1.0 = no offset. Bounded in _generate_day_events.
    basal_dose_drift: float = 1.0
    is_rare_event_day: bool = False
    illness_is_target: float = 1.0
    # Weekday/weekend/holiday tracking
    day_of_week: int = 0               # 0=Monday ... 6=Sunday
    is_holiday: bool = False           # Whether today is a public holiday
    # Time-limited physiological effects
    exercise_is_effects: list = field(default_factory=list)  # (start_idx, end_idx, reduction)
    alcohol_effects: list = field(default_factory=list)      # (start_idx, end_idx, hgo_factor)
    stress_effects: list = field(default_factory=list)       # (start_idx, end_idx, is_factor)
    meal_hgo_effects: list = field(default_factory=list)     # (start_idx, end_idx, magnitude_g_per_hr)
    # Slow physiological state
    glycogen_grams: float = 70.0  # Current hepatic glycogen reserve (g)
    glucotox_bg_ema: float = 120.0  # 3h EMA of true BG, drives glucotoxic IR
    # Hypo correction tracking (see HYPO_CORRECTION_REFRACTORY_MIN).
    last_hypo_correction_idx: int = -9999
    # Active post-hypo basal-suspend envelopes — each entry is
    # (start_idx, until_idx). Multiple can overlap when corrections fire in
    # close succession; the effective factor is the min across active
    # envelopes, which keeps the per-step transition smooth (an extra
    # correction can deepen the suspend but never jumps the factor upward).
    post_hypo_basal_suspend_windows: list = field(default_factory=list)
    # Time index of the next scheduled basal injection. -1 = uninitialised
    # (anchored to the first day's wake_idx on the first _generate_day_events
    # call). Advanced by the fixed once-daily BASAL_DOSE_INTERVAL_HOURS (24h)
    # after each schedule, decoupled from the analogue's action duration.
    next_basal_due_idx: int = -1


# ============================================================================
# CURVE GENERATION UTILITIES
# ============================================================================

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
    # Each element is the amount appearing DURING that step, so integrate the
    # gamma across the step instead of sampling its right edge. Sampling at
    # t = dt, 2dt, ... skips t=0 and makes a curve materialise at a large
    # fraction of its own peak in one step — 27% for a fast meal component, 65%
    # for a rescue — which draws as a stair-step wherever a curve begins and
    # stacks into a sawtooth when several start close together.
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
    """Long-acting basal insulin curve (Bateman one-compartment PK).

    Models subcutaneous basal as first-order absorption + first-order
    elimination: ``f(t) = exp(-ke·t) - exp(-ka·t)``. The result is a smooth
    rise from zero, a broad peak at ``tmax = ln(ka/ke)/(ka-ke)`` (~6.3h for
    the default rates), and a long, gentle post-peak decline matching real
    long-acting analogues (between glargine and degludec). There is no flat
    plateau and no slope discontinuity anywhere. The shape is intentionally
    flatter than glargine PK so that doses scheduled at a cadence of one
    duration-of-action — and thus not overlapping in their high-amplitude
    middles — still produce a near-flat cumulative basal trace.

    A smootherstep window over the last ``tail_clip_hours`` tapers the late
    residual to zero so consecutive daily doses join without a tail-step.
    Normalized so ``sum(values) = total_amount``.

    ``ramp_up_hours`` / ``ramp_down_hours`` are accepted for backward
    compatibility with the legacy trapezoid signature and ignored — the
    shape is now controlled by the PK rate constants.
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

    Subcutaneous insulin DIA scales with dose: larger depots dissolve more
    slowly, peak slightly later, and act for longer. Scaling is centered on a
    5U reference dose so a typical meal bolus matches ``dia_base_hours``.

    ``gamma_k`` / ``gamma_theta`` / ``dia_base_hours`` default to the legacy
    reference bolus but are overridden per patient with the chosen rapid-acting
    analogue's parameters (see BOLUS_VARIANTS).
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
    """Mean-zero daily (dawn-phenomenon) profile for the OU equilibrium.

    A wrapped Gaussian peaking at GE_DAWN_PEAK_HOUR, mean-subtracted over the 24h
    day so the daily component adds rhythm WITHOUT shifting the pooled mean. Scaled
    per-patient by ge_dawn_amplitude in the OU step. Being a deterministic function
    of the hour of day (repeated every day), its variance is day-to-day consistent
    and shows up as 24h autocorrelation and diurnal amplitude — the structure real
    CGM has that opaque OU noise lacks. A smooth Gaussian (a TRANSIENT morning peak)
    beats a sustained daytime plateau here: the slow basal-adjustment feedback
    absorbs a sustained elevation but not a transient dawn rise, so the Gaussian is
    what actually surfaces as visible diurnal amplitude. The 24h mean of the wrapped
    Gaussian is ~width*sqrt(2*pi)/24 (tails negligible for width << 24)."""
    d = ((hour_of_day - GE_DAWN_PEAK_HOUR + 12.0) % 24.0) - 12.0
    raw = float(np.exp(-0.5 * (d / GE_DAWN_WIDTH_HOURS) ** 2))
    daily_mean = GE_DAWN_WIDTH_HOURS * 2.5066282746310002 / 24.0
    return raw - daily_mean


def envelope_intensity(time_idx: int, start_idx: int, end_idx: int,
                       ramp_up_steps: int, ramp_down_steps: int) -> float:
    """Trapezoidal envelope for time-bounded effects.

    Returns 0 outside [start_idx, end_idx). Inside, ramps linearly from 0 to 1
    over ramp_up_steps, plateaus at 1, then ramps back to 0 over ramp_down_steps.
    Used to soften on/off transitions of exercise/stress IS effects and alcohol
    HGO suppression so the BG curves don't show step-function discontinuities.
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
    """Hill-function HGO rate (g/hr) given current plasma insulin per step.

    HGO = SUPPRESSED + (UNSUPPRESSED - SUPPRESSED) / (1 + insulin/HALF_MAX).
    Tuned so a typical basal level (~0.086 U/step under ICR=8) yields exactly
    HGO_BASE (8.25 g/hr), preserving the basal-balances-HGO invariant.
    """
    span = HGO_UNSUPPRESSED_GRAMS_PER_HOUR - HGO_SUPPRESSED_FLOOR_GRAMS_PER_HOUR
    suppression = 1.0 / (1.0 + max(0.0, insulin_per_step) / HGO_INSULIN_HALF_MAX)
    return HGO_SUPPRESSED_FLOOR_GRAMS_PER_HOUR + span * suppression


# ============================================================================
# PATIENT GENERATOR
# ============================================================================

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

    # Physiological parameters — two independent axes (body weight and insulin
    # resistance) widen the population spread enough to span real-T1D TDDs.
    profile.body_weight_kg = float(np.clip(
        rng.normal(BODY_WEIGHT_MEAN_KG, BODY_WEIGHT_SIGMA_KG),
        BODY_WEIGHT_MIN_KG, BODY_WEIGHT_MAX_KG))
    profile.insulin_resistance_factor = float(np.clip(
        np.exp(rng.normal(0.0, IR_LOGNORMAL_SIGMA)),
        IR_FACTOR_MIN, IR_FACTOR_MAX))

    # is_base and ICR are coupled to insulin_resistance_factor (real IR/ICR
    # are physiologically correlated) but each carries a small independent
    # noise term so they aren't perfectly redundant.
    ir = profile.insulin_resistance_factor
    profile.is_base = max(0.3, ir * np.exp(rng.normal(0.0, IR_TO_IS_NOISE_SIGMA)))
    profile.icr = max(3.0, (ICR_MEAN / ir) * np.exp(rng.normal(0.0, IR_TO_ICR_NOISE_SIGMA)))
    profile.correction_factor = max(10.0, rng.normal(CORRECTION_FACTOR_MEAN, CORRECTION_FACTOR_SIGMA) / ir)
    # Per-patient glucose effectiveness (Sg), lognormal around the population
    # mean so real ~2-3x inter-individual spread is preserved; floored so no
    # patient is a pure integrator (which would re-open the long-ACF tail).
    profile.glucose_effectiveness = float(np.clip(
        GE_RATE * np.exp(rng.normal(0.0, GE_REL_SIGMA)),
        GE_RATE_MIN, GE_RATE_MAX))
    # Per-patient equilibrium anchor (the level the stochastic equilibrium
    # reverts to). Its between-patient spread carries the per-patient mean
    # heterogeneity; loosely tied to insulin resistance so resistant patients
    # sit a little higher.
    profile.ge_anchor = float(np.clip(
        rng.normal(GE_EQ_ANCHOR_MEAN + GE_ANCHOR_IR_COUPLING * (ir - 1.0), GE_EQ_ANCHOR_SIGMA),
        110.0, 210.0))
    # Per-patient within-patient-variability amplitude: some patients wander
    # more than others (the global GE_EQ_SIGMA alone made every patient equally
    # variable). Lognormal around 1.0, clipped.
    profile.ge_sigma_mult = float(np.clip(
        np.exp(rng.normal(0.0, GE_SIGMA_REL_SIGMA)),
        GE_SIGMA_MULT_CLIP[0], GE_SIGMA_MULT_CLIP[1]))
    profile.dawn_hgo_amplitude = max(0.0, rng.normal(DAWN_HGO_AMPLITUDE_MEAN, DAWN_HGO_AMPLITUDE_SIGMA))
    profile.night_hgo_dip_amplitude = max(0.0, rng.normal(NIGHT_HGO_DIP_AMPLITUDE_MEAN, NIGHT_HGO_DIP_AMPLITUDE_SIGMA))
    # Equilibrium dawn rhythm scales with the SAME per-patient dawn trait as the HGO
    # surge (one coherent, learnable "dawn strength" per patient): factor 0 (no dawn)
    # to ~2 (strong), so strong-dawn patients get both a bigger HGO surge and a bigger
    # daily equilibrium swing.
    dawn_factor = (min(2.0, profile.dawn_hgo_amplitude / DAWN_HGO_AMPLITUDE_MEAN)
                   if DAWN_HGO_AMPLITUDE_MEAN > 0 else 0.0)
    profile.ge_dawn_amplitude = GE_EQ_DAWN_AMPLITUDE_MEAN * dawn_factor
    profile.exercise_duration_mean_min = float(np.clip(
        rng.normal(EXERCISE_DURATION_MEAN_MIN, EXERCISE_DURATION_MEAN_SIGMA_MIN),
        *EXERCISE_DURATION_MEAN_MIN_CLAMP))

    # Ideal basal balances 24h of HGO at the patient's own insulin sensitivity:
    # at steady state, glucose_out = total_insulin * ICR / IS must equal HGO,
    # so basal = HGO * 24 * IS / ICR. HGO itself is scaled per-patient by
    # body_weight_kg/75 in the step function (heavier liver, more HGO), so the
    # ideal basal must include that factor or heavy patients are under-dosed.
    # Skipping the IS_base or weight factors systematically biases populations.
    # Competent patients (high s3) stay close to ideal; incompetent ones deviate more.
    weight_factor = profile.body_weight_kg / BODY_WEIGHT_MEAN_KG
    ideal_basal = (HGO_BASE_GRAMS_PER_HOUR * 24.0) * weight_factor * profile.is_base / profile.icr
    # Strong nonlinearity on s3 so high-skill patients have near-perfect basal
    # and don't rely on the basal_adjustment feedback (which can oscillate).
    noise_scale = BASAL_DOSE_SIGMA * (1.5 - s3) ** 2.5
    # Clamp widened from [5, 40] to [5, 80] — heavy IR patients can legitimately
    # need 60+ U basal/day (e.g., 110kg patient with IR=1.8).
    profile.basal_dose = float(np.clip(rng.normal(ideal_basal, noise_scale), 5.0, 80.0))

    # Insulin analogue assignment — one rapid-acting bolus and one long-acting
    # basal per patient, drawn uniformly. PK params come from the variant
    # tables; the basal PK action duration replaces the old sampled duration.
    # Both basals are dosed once daily (basal_dose_interval_hours), so the
    # per-injection amount is the full 24h basal_dose (per_dose_factor = 1.0)
    # and the HGO/ICR balance invariant is unchanged.
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
    profile.patience_time_min = (PATIENCE_TIME_INCOMPETENT +
                                  (PATIENCE_TIME_COMPETENT - PATIENCE_TIME_INCOMPETENT) * s3)
    # Quadratic scaling on s3 so the high-skill tail collapses error toward zero
    # (s3=0.95 -> ~9% sigma, s3=0.25 -> ~66% sigma). The linear form gave
    # high-skill patients enough residual error to spend ~30% in TBR.
    profile.carb_count_error_sigma = CARB_COUNT_ERROR_SIGMA_BASE * (1.3 - s3) ** 2
    profile.bolus_timing_mean = (BOLUS_TIMING_COMPETENT_MEAN * s3 +
                                  BOLUS_TIMING_INCOMPETENT_MEAN * (1 - s3))
    profile.bolus_timing_sigma = BOLUS_TIMING_SIGMA_BASE / (0.3 + 0.7 * s3)
    profile.exercise_probability = EXERCISE_PROBABILITY_BASE + EXERCISE_SKILL_BONUS * s4
    profile.panic_factor = HYPO_PANIC_FACTOR_BASE * (1.2 - s3)
    profile.basal_miss_prob = BASAL_MISS_PROB_BASE * np.exp(BASAL_MISS_SKILL_SCALE * (0.5 - s2))
    profile.meal_jitter_sigma_min = MEAL_TIME_JITTER_BASE_MIN / (0.2 + 0.8 * s4)

    profile.wake_time_hours = np.clip(profile.wake_time_hours, 4.0, 12.0)

    return profile


# ============================================================================
# MAIN SIMULATOR
# ============================================================================

class T1DMSimulator:
    """
    Generates factor curves and BG trace for a virtual T1DM patient.
    Call generate() repeatedly to advance the simulation by DT_MINUTES.

    Performance note: curve contributions are pre-accumulated into numpy arrays
    (one per curve type) so that each time step reads contributions in O(1)
    instead of iterating over all active curves. IOB is computed as a numpy
    prefix-sum over the future insulin array.
    """

    def __init__(self, seed: int = 42, initial_bg: Optional[float] = None):
        self.rng = np.random.default_rng(seed)
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

        # Vectorized contribution accumulators (indexed by global step).
        # Curves are scatter-added here on activation so each generate() step
        # reads contributions in O(1) rather than O(n_active_curves).
        _init_len = STEPS_PER_DAY * 4
        self._carb_totals: np.ndarray = np.zeros(_init_len)
        self._basal_totals: np.ndarray = np.zeros(_init_len)
        self._bolus_totals: np.ndarray = np.zeros(_init_len)
        self._exercise_totals: np.ndarray = np.zeros(_init_len)
        self._rescue_totals: np.ndarray = np.zeros(_init_len)  # correction_carb only, for the rule-of-15 recheck

        # EMA-smoothed insulin level used by the HGO Hill function. Models
        # plasma-insulin lag behind subcutaneous absorption.
        self._smoothed_insulin_for_hgo: float = 0.0

        # AR(1) noise state. Each step advances via:
        #   noise = ρ·noise_prev + √(1 − ρ²) · N(0, σ)
        # giving stationary variance σ² but smooth time-correlated noise.
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
        self._rescue_totals = np.zeros(_init_len)  # correction_carb only, for the rule-of-15 recheck
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

        # Daily IS drift state — must be cleared so day 1 doesn't blend with
        # a prior patient's "yesterday" drift (_plan_day reads these via
        # getattr and would otherwise see the previous instance's values).
        for attr in ('_daily_is_drift', '_daily_is_phase_shift',
                     '_prev_daily_is_drift', '_prev_daily_is_phase_shift'):
            if hasattr(self, attr):
                delattr(self, attr)

        self._plan_day()
        self._generate_day_events()

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

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
                # Tracked separately so the rescue branch can see its own
                # carbs-on-board without counting meal carbs, which come with a
                # bolus attached and would make the patient under-treat.
                self._rescue_totals[start_idx:end] += curve
        elif curve_type == 'basal':
            self._basal_totals[start_idx:end] += curve
        elif curve_type in ('bolus', 'insulin'):
            self._bolus_totals[start_idx:end] += curve
        elif curve_type == 'exercise':
            self._exercise_totals[start_idx:end] += curve

    def inject_curve(self, values: np.ndarray, start_idx: int,
                     curve_type: str, label: str = '') -> None:
        """Inject a curve directly into the simulation.

        Use this instead of appending to state.active_curves directly when
        adding curves externally (e.g., in tests or from custom scripts).
        Both the accumulation arrays and active_curves are updated.
        """
        self.state.active_curves.append(ActiveCurve(
            start_time_idx=start_idx,
            values=values,
            curve_type=curve_type,
            label=label
        ))
        self._add_to_totals(values, start_idx, curve_type)

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

        # Daily IS drift — keep yesterday's values so the IS curve blends
        # smoothly across the midnight transition rather than stepping.
        # Drift magnitude scales with (1.5 - s4): consistent-lifestyle patients
        # (sleep, diet, activity) have stabler insulin needs day-to-day; chaotic
        # patients swing more. This is the dominant per-day perturbation, so
        # gating it on s4 is what gives high-skill patients flat BG traces.
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
        # Bedtime = wake + (24 - sleep_hours) so sleep duration is honored.
        # Floor on awake time prevents pathological "wakes up, immediately sleeps".
        awake_hours = max(8.0, 24.0 - sleep_hours)
        sleep_idx = day_start_idx + int((today_wake + awake_hours) * 60 / DT_MINUTES)

        # Store wake/sleep for the day
        self._today_wake_idx = wake_idx
        self._today_sleep_idx = sleep_idx

        # --- Anomalous event flag for the day ---
        anomalous_today = self.rng.random() < ANOMALOUS_EVENT_PROBABILITY
        anomalous_applied = False  # Only apply to first eligible event

        # --- Basal insulin ---
        # Slow basal adjustment based on recent BG history (patient learns over days).
        # Uses a 3-day rolling mean so single bad days don't whipsaw the dose,
        # but persistent over- or under-dosing self-corrects within a couple
        # weeks. Tight dead-band (110-130) around the TIR midpoint — without
        # the lower bound a persistently low-but-not-hypo mean (e.g. 95 mg/dL)
        # never triggers a downward basal adjustment, so sensitive skilled
        # patients spend ~30% in TBR forever.
        basal_adjustment = 1.0
        if len(self.state.bg_history) > 0:
            rolling_window = min(len(self.state.bg_history), 3 * STEPS_PER_DAY)
            recent_bg = self.state.bg_history[-rolling_window:]
            recent_mean = np.mean(recent_bg)
            # Also track a 1-day mean so a single bad day above 220 triggers
            # the extreme-boost without waiting for the 3-day rolling mean to
            # catch up. Real patients react to "yesterday I was stuck high
            # all day", not to a 3-day average.
            one_day_window = min(len(self.state.bg_history), STEPS_PER_DAY)
            one_day_mean = float(np.mean(self.state.bg_history[-one_day_window:]))

            # Dead-band widened from 110-130 to 115-150 in combination with
            # the faster BASAL_DRIFT_ALPHA. The faster alpha helps stuck-high
            # patients recover, but at the old narrow band it also over-corrected
            # mildly-high (130-150) patients, driving them straight into hypo
            # via too-aggressive basal. The wider dead-band leaves the 115-150
            # range alone (real clinicians don't push basal for a 140 mean BG
            # — that's already in TIR), while still aggressively addressing
            # truly hyper (>150) or hypo-leaning (<115) patterns.
            if recent_mean > 150:
                # Skill scales only partially on the upward path — chronically-high
                # patients eventually self-correct regardless of skill (clinician
                # visits, symptoms, fatigue from hyperglycemia). Otherwise low-skill
                # IR patients get stuck above 220 for weeks.
                overshoot = min((recent_mean - 150) / 80.0, 1.0)
                skill_factor = 0.4 + 0.6 * eff_s3   # baseline 40% + up to 100%
                # Extreme-high relief: a patient running high for multiple days is
                # in clinical-emergency territory and behavior shifts (urgent care
                # visit, doctor call, friend/family intervention). Boost the ratio
                # so multi-day hyper streaks recover in 2-3 days rather than 10.
                # Trigger lowered 220→200 so patients stuck in the 200-250 zone
                # (the dominant TAR2 contributor) get basal escalation instead of
                # sitting there until basal-drift catches up.
                trigger_mean = max(recent_mean, one_day_mean)
                if trigger_mean > 200:
                    extreme_boost = 1.0 + 0.5 * min(1.0, (trigger_mean - 200) / 50.0)
                else:
                    extreme_boost = 1.0
                basal_adjustment = 1.0 + overshoot * (BASAL_CORRECTION_MAX_ADJUSTMENT * skill_factor) * extreme_boost
            elif recent_mean < 115:
                # Downward path keeps full skill scaling — low-skill patients
                # tend to ignore mild lows ("just eat something") rather than
                # cut basal.
                undershoot = min((115 - recent_mean) / 50.0, 1.0)
                basal_adjustment = 1.0 - undershoot * (BASAL_CORRECTION_MAX_ADJUSTMENT * eff_s3)

        # Accumulate the reactive adjustment's *delta* into a persistent drift
        # so persistent under/over-dosing shifts the base dose beyond the
        # per-day cap. Real patients (with clinician input) tighten basal in
        # days, not months — at alpha=0.20 a chronically-high patient with
        # daily adjustment=1.22 adds (1.22 - 1.0) * 0.20 = 0.044 per day,
        # reaching the 1.8 mark in ~18 days and 2.5 cap in ~34 days, vs ~45
        # days previously. Combined with the reactive adjustment that's an
        # effective ~120-160% boost — enough to break the most severely
        # under-dosed IR patients out of stuck-high BG within a clinically
        # plausible window.
        BASAL_DRIFT_ALPHA = 0.20
        # Drift cap widened from 1.8 → 2.5 on the upward side: small / IR
        # patients whose initial basal_dose drew near the lower clamp (5 U)
        # could be stuck 30-40% under the ideal even with maximum drift, leading
        # to multi-day hyper streaks. 2.5× headroom lets them recover within a
        # reasonable window. Lower bound stays at 0.5 — no symmetric reason
        # to widen it and it would re-open the sensitive-patient TBR drift.
        s.basal_dose_drift = float(np.clip(
            s.basal_dose_drift + BASAL_DRIFT_ALPHA * (basal_adjustment - 1.0),
            0.5, 1.6))

        # --- Schedule all basals that fall inside today's window ---
        # Real glargine and degludec are dosed once daily, so the injection
        # cadence is basal_dose_interval_hours (24h) regardless of the
        # analogue's PK action duration. Each once-daily dose delivers the
        # full 24h basal_dose (per_dose_factor = 1.0), so 24h-average delivery
        # stays at p.basal_dose / 24h and the HGO/ICR balance is unchanged.
        # The analogue's action_hours (glargine 26h, degludec 42h) drives the
        # PK curve length: degludec's long tail overlaps ~1.75 doses at steady
        # state (very flat), glargine's ~1 dose (mild end-of-day waning).
        day_end_idx = day_start_idx + STEPS_PER_DAY
        basal_interval_steps = max(1, int(p.basal_dose_interval_hours * 60 / DT_MINUTES))
        per_dose_factor = p.basal_dose_interval_hours / 24.0

        if s.next_basal_due_idx < 0:
            # First-ever basal — anchor at today's wake_idx (un-jittered;
            # the per-iteration jitter below adds the ±30 min noise).
            s.next_basal_due_idx = wake_idx

        while s.next_basal_due_idx < day_end_idx:
            jitter_steps = int(self.rng.normal(0, 30) / DT_MINUTES)
            dose_idx = max(self.state.current_idx, s.next_basal_due_idx + jitter_steps)

            if self.rng.random() > p.basal_miss_prob:
                # Administer basal — multiplied by injection-site quality
                dose_noise = 1.0 + self.rng.normal(0, BASAL_DOSE_COMPETENCE_NOISE * (1.2 - eff_s3))
                # Contract site-quality deviation toward 1.0: long-acting basal
                # sites are more absorption-consistent than rapid-acting sites.
                raw_site_q = self._site_quality(eff_s4)
                site_q = 1.0 + (raw_site_q - 1.0) * BASAL_SITE_QUALITY_DAMPING
                actual_dose = max(0.5, p.basal_dose * per_dose_factor
                                  * s.basal_dose_drift * dose_noise
                                  * basal_adjustment * site_q)
                duration = p.basal_duration_hours * 60
                curve = basal_curve(float(actual_dose), duration,
                                    ka_per_hour=p.basal_ka, ke_per_hour=p.basal_ke,
                                    tail_clip_hours=p.basal_tail_clip_hours)
                self._pending_events.append((dose_idx, 'basal', {
                    'curve': curve,
                    'label': (f'Basal {actual_dose:.1f}U '
                              f'{p.basal_type} ({p.basal_duration_hours:.0f}h)'),
                }))

            # Advance whether or not the injection was missed — the patient
            # still intended to inject at this once-daily slot.
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

            # --- Mixed-meal multi-component carbs ---
            # Each meal is composed of 2-5 overlapping gamma absorption curves
            # sampled from fast/medium/slow categories. Component-type weights
            # tilt toward slow with high dietary discipline (s1).
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
                if component_carbs < 0.5:
                    continue
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
                    'label': f'Meal {component_carbs:.0f}g {ctype}'
                }))

            # Protein/fat slow tail — scaled to the meal so snacks don't carry
            # the same 10 g slow tail as a 50 g dinner. Fixed-tail behavior
            # pulled the cohort post-meal envelope peak to ~220 min vs ~100 in
            # real data; this scaling keeps the tail proportional.
            pf_grams = float(np.clip(PROTEIN_FAT_FRACTION_OF_CARBS * carb_amount,
                                     PROTEIN_FAT_MIN_GRAMS, PROTEIN_FAT_MAX_GRAMS))
            pf_curve = gamma_curve(pf_grams, PROTEIN_FAT_GAMMA_K,
                                   PROTEIN_FAT_GAMMA_THETA,
                                   PROTEIN_FAT_GAMMA_K * PROTEIN_FAT_GAMMA_THETA * 4)
            self._pending_events.append((meal_idx, 'carb', {
                'curve': pf_curve, 'label': f'Protein/fat {pf_grams:.0f}g equiv'
            }))

            # Delayed-meal HGO rebound: large meals trigger a positive HGO bump
            # 3.5-5.5h later from delayed gluconeogenesis (amino acids) and cortisol
            # response. This is the mechanism behind nocturnal hyperglycemia
            # after a big dinner.
            if carb_amount > DELAYED_HGO_MEAL_THRESHOLD_GRAMS:
                excess = carb_amount - DELAYED_HGO_MEAL_THRESHOLD_GRAMS
                magnitude = min(DELAYED_HGO_MAX_BUMP, DELAYED_HGO_PER_GRAM * excess)
                delay_h = self.rng.uniform(DELAYED_HGO_DELAY_HOURS_MIN, DELAYED_HGO_DELAY_HOURS_MAX)
                duration_h = self.rng.uniform(DELAYED_HGO_DURATION_HOURS_MIN, DELAYED_HGO_DURATION_HOURS_MAX)
                rebound_start = meal_idx + int(delay_h * 60 / DT_MINUTES)
                rebound_end = rebound_start + int(duration_h * 60 / DT_MINUTES)
                s.meal_hgo_effects.append((rebound_start, rebound_end, magnitude))

            # --- Bolus for this meal ---
            carb_estimate = max(0, carb_amount * (1 + self.rng.normal(
                CARB_COUNT_UNDERBOLUS_BIAS, p.carb_count_error_sigma)))

            # Real pump users bolus for almost everything they eat, including
            # snacks. Skip-prob caps at ~10% for the lowest-skill snackers,
            # and a small main-meal skip rate (~MEAL_BOLUS_SKIP_BASE_PROB) so
            # low-skill patients miss the occasional main-meal bolus too.
            # The main-meal skips are also a source of decorrelating shocks
            # that knock down mid-range BG autocorrelation.
            if i == AFTERNOON_SNACK_SLOT:
                bolus_skip_prob = AFTERNOON_SNACK_BOLUS_SKIP_BASE * (1 - 0.5 * eff_s3)
            elif i >= MEALS_BASE:
                bolus_skip_prob = 0.1 * (1 - eff_s3)
            else:
                bolus_skip_prob = MEAL_BOLUS_SKIP_BASE_PROB * (1 - eff_s3)

            if self.rng.random() > bolus_skip_prob and carb_estimate > 0:
                intended_dose = carb_estimate / p.icr
                bolus_timing_offset = self.rng.normal(p.bolus_timing_mean, p.bolus_timing_sigma)
                bolus_idx = max(self.state.current_idx, meal_idx + int(bolus_timing_offset / DT_MINUTES))

                # PK shape is determined by the intended dose; site quality
                # only modulates the absorbed amount.
                base_k, base_theta, bolus_duration = bolus_pk_for_dose(
                    intended_dose, p.bolus_gamma_k, p.bolus_gamma_theta,
                    p.bolus_dia_base_hours)
                bolus_k = base_k * (1 + self.rng.normal(0, 0.05))
                bolus_theta = base_theta * (1 + self.rng.normal(0, 0.05))
                delivered_dose = intended_dose * self._site_quality(eff_s4)
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

    def _site_quality(self, s4: float) -> float:
        """Per-dose injection site absorption multiplier.

        Patients with low lifestyle_consistency (s4) rotate sites poorly and
        develop lipohypertrophy, leading to higher dose-to-dose variance.
        Returns a multiplier centered on 1.0; values <1 represent poorly
        absorbing scarred sites, >1 the rare hyper-absorbing surge.

        Scaling is super-linear in (1.5 - s4) so high-s4 patients converge
        toward near-perfect absorption — necessary to keep their TBR/TAR low
        once correction frequency goes up (frequent corrections * high site
        variance = overshoot lows).
        """
        sigma = SITE_QUALITY_SIGMA_BASE * (1.5 - s4) ** 1.8
        return float(np.clip(self.rng.normal(1.0, sigma),
                             SITE_QUALITY_MIN, SITE_QUALITY_MAX))

    def _compute_insulin_resistance(self, time_idx: int, active_carb: float = 0.0) -> float:
        """Compute insulin resistance factor at a given time index.

        Includes diurnal pattern, daily drift (smoothed across midnight), illness
        factor, exercise/stress envelopes, glucotoxic IR, postprandial insulin
        resistance, and per-step noise.
        """
        s = self.state

        # Time of day in hours
        hour = (time_idx * DT_MINUTES / 60.0) % 24.0

        # Smooth blend of yesterday's drift/phase into today's over the first
        # IS_DRIFT_TRANSITION_HOURS (smooth-step easing). This prevents the
        # IS curve from stepping at midnight when the daily randoms change.
        if hour < IS_DRIFT_TRANSITION_HOURS:
            raw = hour / IS_DRIFT_TRANSITION_HOURS
            blend = 0.5 - 0.5 * np.cos(raw * np.pi)
            drift = self._prev_daily_is_drift * (1 - blend) + self._daily_is_drift * blend
            phase_shift = self._prev_daily_is_phase_shift * (1 - blend) + self._daily_is_phase_shift * blend
        else:
            drift = self._daily_is_drift
            phase_shift = self._daily_is_phase_shift

        # Diurnal pattern. The returned value is an *insulin
        # resistance* factor (higher = less glucose cleared per unit insulin —
        # see BG-delta formula in generate()). Morning and evening cortisol
        # peaks raise resistance; deep-sleep around 2am lowers it.
        morning = IS_MORNING_AMPLITUDE * np.exp(-0.5 * ((hour - IS_MORNING_PEAK_HOUR - phase_shift) / 2.0) ** 2)
        evening = IS_EVENING_AMPLITUDE * np.exp(-0.5 * ((hour - IS_EVENING_PEAK_HOUR) / 2.5) ** 2)
        night_hour = hour if hour < 12 else hour - 24
        night = -IS_NIGHT_DIP_AMPLITUDE * np.exp(-0.5 * ((night_hour - IS_NIGHT_DIP_HOUR) / 2.0) ** 2)
        diurnal = 1.0 + morning + evening + night

        is_val = self.patient.is_base * diurnal * (1.0 + drift)

        # Illness — always apply the factor. It rests at 1.0 when healthy and
        # ramps smoothly toward 1.0 after recovery (and away from 1.0 at onset),
        # so IS doesn't step at midnight when is_sick toggles.
        is_val *= s.illness_is_factor

        # Post-exercise IS reduction (aerobic exercise increases insulin sensitivity for hours).
        # Trapezoidal envelope softens the on/off edges so IS doesn't step.
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

        # Stress IS effect (transient insulin resistance from cortisol/adrenaline).
        # Envelope blends the factor in/out around 1.0 (no effect).
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

        # Glucotoxicity: sustained hyperglycemia transiently raises IR via the
        # 3h BG EMA. Above GLUCOTOX_BG_THRESHOLD, IR climbs linearly toward
        # GLUCOTOX_MAX_IS_INCREASE at GLUCOTOX_BG_FOR_MAX. Closes a positive
        # feedback loop: high BG → harder to bring down.
        if s.glucotox_bg_ema > GLUCOTOX_BG_THRESHOLD:
            excess = s.glucotox_bg_ema - GLUCOTOX_BG_THRESHOLD
            span = GLUCOTOX_BG_FOR_MAX - GLUCOTOX_BG_THRESHOLD
            intensity = min(1.0, excess / span)
            is_val *= (1.0 + GLUCOTOX_MAX_IS_INCREASE * intensity)

        # Postprandial insulin resistance: T1DM patients lack the incretin /
        # GLP-1 sensitivity boost non-diabetics get with a meal, so the
        # absorbing-carb state is if anything mildly insulin-RESISTANT. While
        # carbs are absorbing, raise IR (insulin clears glucose less effectively).
        # Saturates with active carb load, peaks at POSTPRANDIAL_IR_PENALTY_FACTOR.
        if active_carb > 0.0:
            penalty = POSTPRANDIAL_IR_PENALTY_FACTOR * active_carb / (POSTPRANDIAL_IR_PENALTY_HALF + active_carb)
            is_val *= (1.0 + penalty)

        # Fast noise via AR(1) — same stationary σ as the previous independent
        # draw, but with ~22 min correlation half-life so IS swings are smooth.
        self._ar_is = (NOISE_AR1_RHO_METABOLIC * self._ar_is
                       + NOISE_AR1_INNOV_METABOLIC * self.rng.normal(0, IS_FAST_NOISE_SIGMA))
        is_val *= (1.0 + self._ar_is)

        return max(0.2, is_val)

    def _compute_cgm_observation(self, true_bg: float) -> float:
        """Compute the CGM reading: interstitial lag then proportional noise.

        Interstitial glucose trails plasma by a first-order diffusion lag
        (Rebrin/Steil): ``IG += (BG - IG) * (1 - exp(-dt/tau))`` with
        ``tau = CGM_LAG_MINUTES``. The sensor then reports this delayed-and-
        smoothed value plus AR(1) noise, so it never sees instantaneous BG.
        """
        if CGM_LAG_MINUTES > 0:
            alpha_lag = 1.0 - np.exp(-DT_MINUTES / float(CGM_LAG_MINUTES))
            self._interstitial_bg += alpha_lag * (true_bg - self._interstitial_bg)
            sensed_bg = self._interstitial_bg
        else:
            sensed_bg = true_bg

        # AR(1) sensor noise: real CGMs show smoothly-drifting offsets over
        # 30-60 min windows, not white-noise spikes. The Perlin-like wobble is
        # produced by ρ=0.92 (~42 min half-life) with σ scaled by BG.
        self._ar_cgm = (NOISE_AR1_RHO_SENSOR * self._ar_cgm
                        + NOISE_AR1_INNOV_SENSOR * self.rng.normal(0, CGM_NOISE_FRACTION))
        observed = sensed_bg * (1.0 + self._ar_cgm)
        return np.clip(observed, BG_CLAMP_MIN, BG_CLAMP_MAX)

    def _check_and_correct(self, time_idx: int):
        """Patient checks CGM and possibly corrects highs/lows."""
        p = self.patient
        s = self.state

        # Severe hypo (<55 mg/dL) produces symptoms the patient cannot ignore:
        # sweating, shaking, confusion. Awake patients act immediately; asleep
        # patients wake up. This bypass is what prevents 6+ hour stretches in
        # dangerous hypoglycemia.
        severe_hypo = s.bg_observed < SEVERE_HYPO_THRESHOLD

        is_awake = self._today_wake_idx <= time_idx < self._today_sleep_idx
        if not is_awake:
            if severe_hypo:
                pass  # symptoms wake them — proceed to act this step
            elif s.bg_observed < 55 or s.bg_observed > 350:
                delay_steps = int(self.rng.exponential(HYPO_DETECTION_ASLEEP_LAMBDA) / DT_MINUTES)
                if delay_steps > 0:
                    return
            else:
                return

        # Check interval — bypassed by severe hypo
        steps_since_check = time_idx - s.last_cgm_check_idx
        check_interval_steps = int(p.cgm_check_interval_min / DT_MINUTES)
        if steps_since_check < check_interval_steps and not severe_hypo:
            return

        s.last_cgm_check_idx = time_idx

        # Compute insulin on board (IOB) from the pre-accumulated insulin array.
        # This is O(n_future) with numpy, faster than iterating active_curves.
        if time_idx < len(self._bolus_totals):
            iob = float(np.sum(self._bolus_totals[time_idx:]))
        else:
            iob = 0.0

        # Skill-scaled correction thresholds: attentive/competent patients act on
        # smaller excursions while unskilled patients tolerate more excursion
        # before acting. Mild offsets — large offsets caused skilled patients to
        # over-correct frequently and rebound into hypo.
        skill_avg = (p.attentiveness + p.dosing_competence) / 2.0
        # Effective thresholds. The low-side skill multiplier (18) is chosen
        # so attentive patients catch BG drops *before* crossing 70 rather
        # than reactively after. For skill_avg=0.7 the trigger lands at 73 —
        # still below the TIR midpoint, but enough to
        # cover the ~1-step lag between detection and rescue carb.
        eff_low_thresh = p.hypo_threshold
        eff_high_thresh = BG_HIGH_THRESHOLD - 25.0 * skill_avg

        # --- Handle hypoglycemia ---
        if s.bg_observed < eff_low_thresh:
            # Refractory: in moderate hypo (55-70) a second correction within
            # 20 min just stacks carbs on top of carbs that haven't acted yet.
            # Severe hypo (<55) uses a SHORTER refractory (10 min) rather than
            # bypassing entirely — symptomatic patients still rage-eat multiple
            # times if BG stays low, but rule-of-15 says wait between doses so
            # the first rescue's carbs can start acting. Without the gap the
            # CGM-check bypass let the patient stack 3-5 rage doses in 10-15 min,
            # then overshoot to 140-180 (visible sawtooth). A full bypass also
            # broke TBR2 when stacked carbs produced post-correction crashes.
            refractory_min = SEVERE_HYPO_REFRACTORY_MIN if severe_hypo else HYPO_CORRECTION_REFRACTORY_MIN
            refractory_steps = int(refractory_min / DT_MINUTES)
            if time_idx - s.last_hypo_correction_idx < refractory_steps:
                return

            # Rule-of-15 recheck: account for rescue glucose already swallowed
            # and still absorbing before deciding whether to eat more, and how
            # much. Skill-scaled, and a rage-eat roll — likelier the less
            # competent the patient — discards the arithmetic entirely; a
            # frightened patient at 40 eats the whole packet.
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

            severity = max(0, eff_low_thresh - projected_bg)
            # Skilled patients eat more carbs (toward classical rule-of-15) so
            # they recover from over-bolus crashes; unskilled under-correct and
            # linger in hypo.
            skill_grams_multiplier = 1.0 + 1.5 * skill_avg
            correction_grams = (HYPO_CORRECTION_BASE_GRAMS * skill_grams_multiplier
                                + p.panic_factor * severity / 20.0)

            # Severe hypo (<55) is symptomatic — patient rage-eats reflexively,
            # not probabilistically. The grams floor scales with how far below
            # 55 the BG is. Tuned so a BG=30 → ~22g (clears severe in 15-30 min
            # without overshooting clean past 70 to TIR), not 41g (which would
            # collapse total TBR by ejecting the patient straight to hyper).
            if severe_hypo:
                deficit = max(0.0, SEVERE_HYPO_THRESHOLD - projected_bg)
                correction_grams = max(correction_grams, 14.0 + 0.35 * deficit)
            # (The former probabilistic rage-eat branch below RAGE_EAT_BG_THRESHOLD=50
            # was unreachable — any BG < 50 already trips severe_hypo (< 55) above,
            # which rescues deterministically. Reflexive severe-hypo eating supersedes
            # the stochastic version.)

            # Hypo correction uses fast-acting carbs (glucose tablets / juice)
            k = HYPO_CARB_K
            theta = HYPO_CARB_THETA
            duration = max(k * theta * 4, 60)
            curve = gamma_curve(correction_grams, k, theta, duration)
            self.inject_curve(curve, time_idx, 'correction_carb',
                              f'Hypo correction {correction_grams:.0f}g')
            # Slow-carb follow-up snack: standard clinical advice after a
            # rescue is "treat the low, then eat a small carb+protein snack"
            # to keep glucose coming while the active bolus continues to act.
            # Without this the sawtooth recurs: BG climbs from the fast carbs,
            # then drops again 60-90 min later when fast carbs are gone but
            # the meal bolus is still working. Skill-gated — only attentive
            # patients remember the follow-up; low-skill patients just eat the
            # fast carbs and re-hypo. Amount scales with the rescue dose so
            # severe-hypo cascades get a proportional tail.
            if skill_avg > HYPO_FOLLOWUP_SKILL_THRESHOLD:
                followup_grams = correction_grams * HYPO_FOLLOWUP_FRACTION
                fk = HYPO_FOLLOWUP_GAMMA_K
                ft = HYPO_FOLLOWUP_GAMMA_THETA
                followup_curve = gamma_curve(followup_grams, fk, ft, fk * ft * 4)
                self.inject_curve(followup_curve, time_idx, 'correction_carb',
                                  f'Hypo followup {followup_grams:.0f}g slow')
            s.last_hypo_correction_idx = time_idx

            # Scale down basal insulin for the next ~90 min after any hypo
            # correction (awake or asleep). Real patients respond to a hypo by
            # also reducing future basal coverage — pump suspend / temp basal
            # for pump users, skipping the next basal injection for MDI — not
            # just by eating. Without this scale-down the forward basal pipeline
            # keeps clearing glucose as fast as the patient eats, producing a
            # sawtooth of repeated snacks that never bring BG back to range.
            suspend_steps = int(POST_HYPO_BASAL_SUSPEND_DURATION_HOURS * 60 / DT_MINUTES)
            # Each correction adds a fresh sin² envelope anchored at the
            # current step. Overlapping envelopes are combined by taking the
            # min factor at each step, so the transition stays smooth no
            # matter how often corrections fire.
            s.post_hypo_basal_suspend_windows.append(
                (time_idx, time_idx + suspend_steps))

            # After a SEVERE hypo correction, recheck soon (don't wait the full
            # CGM interval). Mild hypos keep the normal cadence so they linger
            # naturally — what we're killing here is the dangerous tail, not
            # all sub-70 time.
            if severe_hypo:
                recheck_steps = max(1, 15 // DT_MINUTES)
                s.last_cgm_check_idx = time_idx - check_interval_steps + recheck_steps

        # --- Handle hyperglycemia ---
        elif s.bg_observed > eff_high_thresh:
            steps_since_correction = time_idx - s.last_correction_idx
            # Urgency now ramps from BG_HIGH_THRESHOLD (180) up. At 180 → 1.0
            # (full patience), at 230 → 2.0 (half), at 280 → 3.0 (third). Without
            # this, sustained 200-250 BG sat for the full 2h patience window and
            # produced the heavy >250 tail (TAR2 15% vs Ohio 9%).
            urgency = min(3.0, 1.0 + max(0.0, (s.bg_observed - BG_HIGH_THRESHOLD) / 50.0))
            patience_steps = int(p.patience_time_min / (DT_MINUTES * urgency))

            if steps_since_correction >= patience_steps:
                # IOB-aware correction: subtract a baseline 70% of expected IOB drop for
                # everyone (real pump users always see a "remaining active insulin" estimate)
                # plus a skill-scaled bonus up to 30%. Without the baseline floor, low-skill
                # patients stacked corrections and crashed (TBR ~10% against OhioT1DM's ~3%).
                iob_equiv_bg_drop = iob * p.correction_factor
                iob_consideration = iob_equiv_bg_drop * (0.7 + 0.3 * p.dosing_competence)
                adjusted_excess = max(0.0, (s.bg_observed - BG_TARGET) - iob_consideration)
                # Only correct when there is genuine excess above the IOB-aware
                # target. Skilled patients have eff_high_thresh dip below BG_TARGET
                # (175 - 25*skill_avg < 158 for skill_avg > 0.68), so this branch can
                # fire with adjusted_excess == 0; without this guard the max(0.5, ..)
                # floor would inject a spurious 0.5U bolus into an already-in-range BG.
                if adjusted_excess > 0.0:
                    correction_dose = adjusted_excess / p.correction_factor
                    correction_dose *= (1 + self.rng.normal(0, p.carb_count_error_sigma * 0.5))
                    correction_dose = max(0.5, correction_dose)

                    if s.bg_observed > RAGE_BOLUS_BG_THRESHOLD:
                        rage_prob = RAGE_BOLUS_PROBABILITY_BASE * (1.2 - p.dosing_competence)
                        if self.rng.random() < rage_prob:
                            rage_mult = self.rng.uniform(RAGE_BOLUS_MULTIPLIER_MIN, RAGE_BOLUS_MULTIPLIER_MAX)
                            correction_dose *= rage_mult

                    base_k, base_theta, corr_duration = bolus_pk_for_dose(
                        correction_dose, p.bolus_gamma_k, p.bolus_gamma_theta,
                        p.bolus_dia_base_hours)
                    delivered_dose = correction_dose * self._site_quality(p.lifestyle_consistency)
                    bolus_curve = gamma_curve(delivered_dose, base_k, base_theta, corr_duration)
                    self.inject_curve(bolus_curve, time_idx, 'bolus',
                                      f'Correction {delivered_dose:.1f}U')
                    s.last_correction_idx = time_idx

        # --- Trend-based anticipatory corrections ---
        elif len(s.bg_obs_history) >= TREND_CORRECTION_WINDOW_STEPS:
            steps_since_correction = time_idx - s.last_correction_idx
            patience_steps = int(p.patience_time_min / DT_MINUTES)
            if steps_since_correction >= patience_steps:
                # Slope from the CGM-observed history (bg_obs_history), not true BG:
                # the patient can only act on what the sensor shows.
                window = s.bg_obs_history[-TREND_CORRECTION_WINDOW_STEPS:]
                trend = (window[-1] - window[0]) / (TREND_CORRECTION_WINDOW_STEPS - 1)

                if (trend > TREND_HIGH_RATE_THRESHOLD and
                        s.bg_observed > TREND_HIGH_BG_MIN and
                        s.bg_observed <= eff_high_thresh):
                    if self.rng.random() < p.attentiveness:
                        projected_rise = trend * TREND_CORRECTION_WINDOW_STEPS * 2
                        correction_dose = max(0.5, projected_rise * p.attentiveness / p.correction_factor)
                        base_k, base_theta, corr_duration = bolus_pk_for_dose(
                            correction_dose, p.bolus_gamma_k, p.bolus_gamma_theta,
                            p.bolus_dia_base_hours)
                        delivered_dose = correction_dose * self._site_quality(p.lifestyle_consistency)
                        bolus_curve = gamma_curve(delivered_dose, base_k, base_theta, corr_duration)
                        self.inject_curve(bolus_curve, time_idx, 'bolus',
                                          f'Trend corr {delivered_dose:.1f}U')
                        s.last_correction_idx = time_idx

                elif (trend < TREND_LOW_RATE_THRESHOLD and
                          s.bg_observed < TREND_LOW_BG_MAX and
                          s.bg_observed >= eff_low_thresh):
                    # Honor the hypo-correction refractory — otherwise a trend-low
                    # snack and a regular hypo correction can fire ~5 min apart
                    # on the same dipping BG, double-stacking carbs and creating
                    # post-correction overshoots.
                    refractory_steps = int(HYPO_CORRECTION_REFRACTORY_MIN / DT_MINUTES)
                    if time_idx - s.last_hypo_correction_idx < refractory_steps:
                        return
                    if self.rng.random() < p.attentiveness:
                        correction_grams = float(np.clip(
                            abs(trend) * TREND_CORRECTION_WINDOW_STEPS * 2.0, 5.0, 20.0))
                        # Pre-emptive low correction uses fast-acting carbs
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

            # BG-aware meal-bolus gating. A scheduled pre-bolus is the dominant
            # sawtooth driver: patient corrects a hypo, climbs briefly above 70,
            # the meal pre-bolus fires anyway, BG dives again. Real T1Ds glance
            # at the CGM and skip / reduce when low.
            if event_type == 'bolus':
                check_prob = BOLUS_BG_CHECK_BASE_PROB + 0.05 * p.attentiveness
                if self.rng.random() < check_prob:
                    bg = s.bg_observed
                    if bg < p.hypo_threshold:
                        continue  # treat hypo first — meal carbs alone
                    elif bg < p.hypo_threshold + BOLUS_REDUCE_MARGIN:
                        scale = BOLUS_REDUCE_FACTOR_BASE + 0.3 * p.dosing_competence
                        curve = curve * scale
                        label = f"{label} (low-BG reduced ×{scale:.2f})"

            # Same glance at the CGM before a planned session. Unconditional
            # rather than probabilistic like the bolus: setting off on a workout
            # is a deliberate act with preparation around it, not the habitual
            # reflex a mealtime dose is. Skipping here also skips the
            # post-exercise IS boost below, which is correct — no session, no
            # sensitivity tail.
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
        basal_step = float(self._basal_totals[idx]) if idx < len(self._basal_totals) else 0.0
        bolus_step = float(self._bolus_totals[idx]) if idx < len(self._bolus_totals) else 0.0
        # Post-hypo basal stand-down (set by _check_cgm_and_correct after any
        # hypo correction). Represents pump suspend / next-basal skip behavior.
        # Each correction queues a sin² envelope over the suspend window;
        # multiple overlapping envelopes are combined by taking the minimum
        # factor at each step. The sin² profile is C∞-smooth (no slope
        # discontinuities anywhere) with max slope π / window — over a 4h
        # window with depth 0.50 the worst 10-min change is ~6.5%. The
        # min-over-envelopes combiner means a fresh correction during an
        # ongoing suspend deepens or extends the dip without ever jumping
        # the factor upward.
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

        # Per-step absorption noise via AR(1) (smooth, correlated — gut motility
        # and SC depot uptake don't reset every 5 min). Stationary σ matches the
        # original NOISE_SIGMA constants; AR(1) reduces step-to-step jaggedness
        # without changing total spread.
        self._ar_carb = (NOISE_AR1_RHO_METABOLIC * self._ar_carb
                         + NOISE_AR1_INNOV_METABOLIC * self.rng.normal(0, CARB_ABSORPTION_NOISE_SIGMA))
        self._ar_insulin = (NOISE_AR1_RHO_METABOLIC * self._ar_insulin
                            + NOISE_AR1_INNOV_METABOLIC * self.rng.normal(0, INSULIN_ABSORPTION_NOISE_SIGMA))
        if total_carb > 0.0:
            total_carb = max(0.0, total_carb * (1.0 + self._ar_carb))
        if total_insulin > 0.0:
            insulin_noise = 1.0 + self._ar_insulin
            total_insulin = max(0.0, total_insulin * insulin_noise)
            basal_step = max(0.0, basal_step * insulin_noise)
            bolus_step = max(0.0, bolus_step * insulin_noise)

        # --- HGO with insulin-mediated suppression (Hill function) ---
        # Plasma insulin lags subcutaneous absorption, so feed an EMA-smoothed
        # insulin level into the Hill function. This prevents HGO from stepping
        # at the moment a new bolus curve activates (insulin's first non-zero
        # step would otherwise instantly drop HGO).
        self._smoothed_insulin_for_hgo = (
            HGO_INSULIN_SMOOTHING_ALPHA * total_insulin
            + (1.0 - HGO_INSULIN_SMOOTHING_ALPHA) * self._smoothed_insulin_for_hgo
        )
        self._ar_hgo = (NOISE_AR1_RHO_METABOLIC * self._ar_hgo
                        + NOISE_AR1_INNOV_METABOLIC * self.rng.normal(0, HGO_NOISE_SIGMA))
        hgo_rate = compute_hgo_rate(self._smoothed_insulin_for_hgo) * (1 + self._ar_hgo)
        hgo_value = hgo_rate * (DT_MINUTES / 60.0)
        # Scale HGO by body weight (heavier liver, proportionally more endogenous
        # glucose). The basal calibration in generate_patient mirrors this scale,
        # so the structural HGO-balances-basal invariant holds across weights.
        hgo_value *= self.patient.body_weight_kg / BODY_WEIGHT_MEAN_KG

        # Circadian HGO modulation — cortisol-driven dawn surge (~6:30am) plus a
        # deep-sleep trough (~3am). Added to hgo_value in g/hr rather than as a
        # multiplier so the surge isn't fully cancelled by basal insulin
        # coverage (the Hill suppression already shrinks the multiplicative
        # form to ~0). This is what produces the dawn phenomenon visible in
        # real CGM data. Wraps around midnight via the night_hour shift used
        # in the IS diurnal block.
        hour_of_day = (idx * DT_MINUTES / 60.0) % 24.0
        night_h = hour_of_day if hour_of_day < 12 else hour_of_day - 24
        dawn_g_per_hr = self.patient.dawn_hgo_amplitude * np.exp(
            -0.5 * ((hour_of_day - DAWN_HGO_PEAK_HOUR) / DAWN_HGO_SIGMA_HOURS) ** 2)
        night_dip_g_per_hr = self.patient.night_hgo_dip_amplitude * np.exp(
            -0.5 * ((night_h - NIGHT_HGO_DIP_HOUR) / NIGHT_HGO_DIP_SIGMA_HOURS) ** 2)
        hgo_value += (dawn_g_per_hr - night_dip_g_per_hr) * (DT_MINUTES / 60.0)
        hgo_value = max(0.0, hgo_value)

        # Glycogen reservoir gating: when the liver runs low, glycogenolysis can't
        # sustain HGO and total output drops toward the gluconeogenesis-only floor.
        glycogen_low_threshold = GLYCOGEN_CAPACITY_GRAMS * GLYCOGEN_LOW_THRESHOLD_FRACTION
        if s.glycogen_grams < glycogen_low_threshold:
            # Linear scaling on the glycogenolysis-sourced fraction of HGO
            availability = max(0.0, s.glycogen_grams / glycogen_low_threshold)
            hgo_value *= (1.0 - GLYCOGEN_DRAIN_FRACTION) + GLYCOGEN_DRAIN_FRACTION * availability

        # Alcohol additionally suppresses HGO (gluconeogenesis blockade).
        # Trapezoidal envelope around 1.0 (no effect) prevents the HGO curve
        # from stepping at the start/end of an alcohol session.
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

        # Delayed-meal HGO rebound: large meals trigger a delayed positive HGO
        # bump 3.5-5.5h later. Trapezoidal envelope around 0 (no effect).
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

        # Glycogen drain (by the glycogenolysis-sourced fraction of HGO) and
        # refill from absorbed carbs. Tracked as a background reservoir — does
        # not subtract from BG-bound carbs (ICR is empirically tuned to net BG
        # response, so adding a "leak to glycogen" would double-count). The
        # gating above is what couples glycogen back to BG dynamics.
        s.glycogen_grams -= hgo_value * GLYCOGEN_DRAIN_FRACTION
        s.glycogen_grams += total_carb * GLYCOGEN_REFILL_FRACTION
        s.glycogen_grams = float(np.clip(s.glycogen_grams, 0.0, GLYCOGEN_CAPACITY_GRAMS))

        # Remove expired entries from active_curves (memory management for external consumers)
        s.active_curves = [c for c in s.active_curves
                           if (idx - c.start_time_idx) < len(c.values)]

        # --- Insulin sensitivity (modulates insulin effectiveness, not carb load) ---
        insulin_resistance_factor = self._compute_insulin_resistance(idx, active_carb=total_carb)

        # --- Compute BG delta ---
        # IS now divides insulin's effect: resistant patients (IR>1) clear less
        # glucose per unit insulin; sensitive patients (IR<1) clear more.
        glucose_in = total_carb + hgo_value - total_exercise
        glucose_out = total_insulin * p.icr / insulin_resistance_factor
        bg_delta = BG_SCALE_FACTOR * (glucose_in - glucose_out)

        # Glucose effectiveness (Bergman Sg): insulin-independent restoring pull
        # toward a STOCHASTIC equilibrium E(t). E is an Ornstein-Uhlenbeck
        # process — it mean-reverts (timescale GE_EQ_TAU_HOURS) toward the
        # patient's anchor plus a diurnal lift, perturbed each step by Gaussian
        # noise of stationary std GE_EQ_SIGMA. E's own timescale, not the strength
        # of the Sg pull, is what keeps the 8h ACF near zero; Sg is deliberately
        # weak (see GE_RATE) so it does not high-pass slow inputs such as insulin.
        # E's wandering adds distributional spread but decorrelates within hours,
        # so it does not re-inject long-lag autocorrelation.
        ge_mu = p.ge_anchor + p.ge_dawn_amplitude * ge_diurnal_profile(hour_of_day)
        ge_rho = float(np.exp(-DT_MINUTES / (GE_EQ_TAU_HOURS * 60.0)))
        self._ge_equilibrium = (
            ge_mu + ge_rho * (self._ge_equilibrium - ge_mu)
            + np.sqrt(1.0 - ge_rho * ge_rho) * GE_EQ_SIGMA * p.ge_sigma_mult * self.rng.normal())
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

        # Soft-bound headroom cap: if a step would carry BG past the soft
        # threshold toward the hard clamp, cap the move to a fraction of the
        # remaining headroom from the *current* position. BG decays geometrically
        # toward the bound (e.g. with fraction=0.3, gap halves every ~2 steps),
        # so it asymptotes smoothly instead of slamming into a flat line.
        # Checking the projected position (not current) catches large single-step
        # deltas that would otherwise leap clear over the soft zone.
        if bg_delta < 0:
            projected = s.bg + bg_delta
            if projected < BG_SOFT_FLOOR:
                headroom = max(0.0, s.bg - BG_CLAMP_MIN)
                bg_delta = max(bg_delta, -SOFT_APPROACH_FRACTION * headroom)
        elif bg_delta > 0:
            projected = s.bg + bg_delta
            if projected > BG_SOFT_CEILING:
                headroom = max(0.0, BG_CLAMP_MAX - s.bg)
                bg_delta = min(bg_delta, SOFT_APPROACH_FRACTION * headroom)

        # Hard clamp as absolute backstop (should rarely fire)
        s.bg = float(np.clip(s.bg + bg_delta, BG_CLAMP_MIN, BG_CLAMP_MAX))

        # Update glucotoxicity BG EMA (slow, ~3h half-life). Drives transient
        # IR when chronically elevated.
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
            'total_carb': [], 'total_insulin': [], 'basal_insulin': [],
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
            'patience_time': f'{p.patience_time_min:.0f}min',
            'exercise_prob': f'{p.exercise_probability:.2f}',
            'basal_miss_prob': f'{p.basal_miss_prob:.4f}',
            'slow_carb_pref': f'{p.slow_carb_preference:.2f}',
            'panic_factor': f'{p.panic_factor:.2f}',
        }


# ============================================================================
# QUICK TEST
# ============================================================================

if __name__ == '__main__':
    sim = T1DMSimulator(seed=42)
    print("Patient:", sim.get_patient_summary())
    data = sim.generate_hours(24)
    print(f"24h BG range: {data['bg'].min():.0f} - {data['bg'].max():.0f} mg/dL")
    print(f"Mean BG: {data['bg'].mean():.0f} mg/dL")
    print(f"Steps: {len(data['bg'])}")
