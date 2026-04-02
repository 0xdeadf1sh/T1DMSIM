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

# Skill correlation
SKILL_CORRELATION = 0.7  # Off-diagonal correlation in skill covariance matrix
SKILL_VARIANCE = 0.5  # Lower = more patients near average, fewer extremes
SKILL_MIN = 0.55  # Lowest possible skill level (0 = no skill)
SKILL_MAX = 0.95  # Highest possible skill level (1 = perfect)

# Wake/sleep
WAKE_TIME_MEAN_HOURS = 8.0  # Mean wake time (hours from midnight)
WAKE_TIME_SIGMA_BASE = 0.5  # Base sigma for wake time (hours), scaled by 1/s4
SLEEP_DURATION_MEAN_HOURS = 7.5
SLEEP_DURATION_SIGMA_HOURS = 1.0

# Meals
MEALS_BASE = 3  # Base number of meals per day
MEALS_EXTRA_LAMBDA = 2.0  # Extra meals Poisson lambda, scaled by (1 - s1)
MEAL_TIME_OFFSETS_HOURS = [0.5, 5.0, 11.0]  # Breakfast, lunch, dinner offset from wake
MEAL_TIME_JITTER_BASE_MIN = 15.0  # Base jitter in minutes, scaled by 1/s4
MEAL_CARB_MEANS = [40.0, 55.0, 65.0]  # Mean carbs (g) per meal slot
MEAL_CARB_SIGMA = 15.0  # Sigma for carb amount
MEAL_CARB_DISCIPLINE_SCALE = 0.7  # How much s1 reduces carb intake
SNACK_CARB_MEAN = 20.0
SNACK_CARB_SIGMA = 10.0

# Protein/fat baseline (modeled as ~10g slow carbs per meal even for zero-carb meals)
PROTEIN_FAT_EQUIV_GRAMS = 10.0  # Equivalent grams of very slow carbs from protein/fat
PROTEIN_FAT_GAMMA_K = 6.0  # Very slow absorption
PROTEIN_FAT_GAMMA_THETA = 30.0  # Peak at ~180 min

# Carb curve (gamma distribution parameters)
FAST_CARB_K = 2.0  # Gamma shape for fast carbs
FAST_CARB_THETA = 15.0  # Gamma scale for fast carbs (minutes)
SLOW_CARB_K = 4.0  # Gamma shape for slow carbs
SLOW_CARB_THETA = 20.0  # Gamma scale for slow carbs (minutes)
SLOW_CARB_PREFERENCE_BASE = 0.3  # Base probability of choosing slow carbs
SLOW_CARB_PREFERENCE_SKILL_BONUS = 0.5  # Added probability from s1

# Carb curve noise
CARB_CURVE_K_NOISE = 0.1  # Relative noise on gamma k
CARB_CURVE_THETA_NOISE = 0.1  # Relative noise on gamma theta

# Insulin sensitivity
IS_BASE_MEAN = 1.0
IS_BASE_SIGMA = 0.2
IS_DAILY_DRIFT_SIGMA = 0.05  # Day-to-day drift
IS_FAST_NOISE_SIGMA = 0.02  # Step-to-step noise
IS_DAWN_PHASE_DAILY_SIGMA = 1.5  # Hours of day-to-day variation in dawn phenomenon timing

# Insulin sensitivity diurnal components (multiple peaks)
IS_MORNING_PEAK_HOUR = 7.0    # Morning resistance peak
IS_MORNING_AMPLITUDE = 0.25   # Strength of morning resistance
IS_EVENING_PEAK_HOUR = 20.0   # Evening resistance peak
IS_EVENING_AMPLITUDE = 0.20   # Strength of evening resistance
IS_NIGHT_DIP_HOUR = 2.0       # Nighttime sensitivity peak (low resistance)
IS_NIGHT_DIP_AMPLITUDE = 0.15 # How much more sensitive at night

# Illness
ILLNESS_PROBABILITY_BASE = 0.01  # Per-day probability of getting sick
ILLNESS_HEALTH_WEIGHT = 0.3  # How much s4 reduces illness probability
ILLNESS_DURATION_MIN_DAYS = 2
ILLNESS_RECOVERY_PROB = 0.3  # Geometric distribution parameter
ILLNESS_IS_FACTOR_MIN = 1.1
ILLNESS_IS_FACTOR_MAX = 2.0
ILLNESS_IS_RAMP_RATE = 0.1  # How fast illness IS factor changes per day (0 to 1)

# Basal insulin (long-acting)
# Note: ideal basal dose is derived from HGO and ICR in generate_patient().
BASAL_DOSE_SIGMA = 2.0  # Sigma around the HGO/ICR-derived ideal dose
BASAL_DOSE_COMPETENCE_NOISE = 0.08  # Relative noise, scaled by 1/s3
BASAL_DURATION_HOURS = 28.0  # Duration of action
BASAL_MISS_PROB_BASE = 0.02  # Base probability of missing basal dose
BASAL_MISS_SKILL_SCALE = 5.0  # How much skills reduce miss probability
BASAL_CORRECTION_MAX_ADJUSTMENT = 0.50  # Max % a patient will adjust basal in one day
BASAL_RAMP_UP_HOURS = 3.0 # How long it will take before basal insulin peaks in the bloodstream
BASAL_RAMP_DOWN_HOURS = 4.0 # How long it will take before basal insulin decays completely (from peak)

# Bolus insulin (rapid-acting)
BOLUS_GAMMA_K = 3.0
BOLUS_GAMMA_THETA = 25.0  # Peak around 60 min
BOLUS_DURATION_HOURS = 5.0
ICR_MEAN = 10.0  # Insulin-to-carb ratio (1 unit per X grams)
ICR_SIGMA = 2.0
BOLUS_TIMING_COMPETENT_MEAN = -20.0  # Minutes before meal (negative = before)
BOLUS_TIMING_INCOMPETENT_MEAN = 15.0  # Minutes after meal
BOLUS_TIMING_SIGMA_BASE = 5.0  # Base timing variance

# Carb counting error
CARB_COUNT_ERROR_SIGMA_BASE = 0.2  # Relative error, scaled by 1/s3

# Insulin stacking
CGM_CHECK_INTERVAL_ATTENTIVE = 15  # Minutes between checks for attentive patient
CGM_CHECK_INTERVAL_INATTENTIVE = 120  # Minutes for inattentive patient
PATIENCE_TIME_COMPETENT = 180  # Minutes before re-correcting (competent)
PATIENCE_TIME_INCOMPETENT = 30  # Minutes before re-correcting (incompetent)
CORRECTION_FACTOR_MEAN = 40.0  # mg/dL drop per unit of insulin
CORRECTION_FACTOR_SIGMA = 10.0
BG_TARGET = 100.0  # Target BG for corrections
BG_HIGH_THRESHOLD = 180.0  # Threshold to trigger correction
BG_LOW_THRESHOLD = 70.0  # Threshold for hypo correction

# Hypo correction
HYPO_CORRECTION_BASE_GRAMS = 15.0  # Base correction (rule of 15)
HYPO_PANIC_FACTOR_BASE = 2.0  # How much extra is eaten, scaled by 1/s3
HYPO_DETECTION_AWAKE_MINUTES = 5.0  # Detection delay awake
HYPO_DETECTION_ASLEEP_LAMBDA = 30.0  # Exponential mean for detection delay asleep

# Exercise
EXERCISE_PROBABILITY_BASE = 0.3  # Base daily probability
EXERCISE_SKILL_BONUS = 0.4  # Added probability from s4
EXERCISE_TIME_MEAN_OFFSET_HOURS = 9.0  # Typical time: wake + 9h (afternoon/evening)
EXERCISE_TIME_SIGMA_HOURS = 2.0
EXERCISE_DURATION_MEAN_MIN = 40.0
EXERCISE_DURATION_SIGMA_MIN = 15.0
EXERCISE_CARB_EQUIV_PER_MIN = 0.5  # Negative carb equivalent per minute of exercise
EXERCISE_GAMMA_K = 3.0
EXERCISE_GAMMA_THETA = 15.0

# Hepatic glucose output
HGO_BASE_GRAMS_PER_HOUR = 9.0  # ~1.5-2 mg/kg/min for 70kg person ≈ 9g/hr
HGO_NOISE_SIGMA = 0.05  # Relative noise

# BG computation
BG_SCALE_FACTOR = 4.0  # Alpha: converts abstract units to mg/dL per step
BG_CLAMP_MIN = 40.0
BG_CLAMP_MAX = 500.0
BG_INITIAL_MEAN = 120.0
BG_INITIAL_SIGMA = 30.0

# BG regulatory computation
RENAL_THRESHOLD = 180.0  # Kidneys start excreting glucose above this
RENAL_CLEARANCE_RATE = 0.005  # Fraction of excess BG cleared per step
COUNTER_REGULATORY_THRESHOLD = 70.0  # Body releases glucagon below this
COUNTER_REGULATORY_RATE = 2.0  # mg/dL added per step when below threshold

# CGM noise
CGM_LAG_MINUTES = 10  # Interstitial delay
CGM_NOISE_FRACTION = 0.01  # ~1% MARD

# Rare events
RARE_EVENT_PROBABILITY = 0.02  # Per-day probability of a rare/chaotic day
RARE_EVENT_SKILL_REDUCTION = 0.3  # Even skilled people have bad days sometimes

# Rage behavior
RAGE_EAT_BG_THRESHOLD = 55.0       # Below this, patient may rage eat
RAGE_EAT_CARB_MIN = 40.0           # Minimum rage eat carbs
RAGE_EAT_CARB_MAX = 100.0          # Maximum rage eat carbs
RAGE_EAT_PROBABILITY_BASE = 0.3    # Base chance of rage eating when below threshold
RAGE_BOLUS_BG_THRESHOLD = 300.0    # Above this, patient may rage bolus
RAGE_BOLUS_MULTIPLIER_MIN = 1.5    # Minimum dose multiplier during rage bolus
RAGE_BOLUS_MULTIPLIER_MAX = 3.0    # Maximum dose multiplier during rage bolus
RAGE_BOLUS_PROBABILITY_BASE = 0.3  # Base chance of rage bolusing when above threshold

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
EXERCISE_IS_DURATION_HOURS = 18.0      # Duration of post-exercise IS boost (hours)

# ============================================================================
# TREND-BASED ANTICIPATORY CORRECTIONS
# ============================================================================

TREND_CORRECTION_WINDOW_STEPS = 6      # BG history window for trend (6 steps = 30 min)
TREND_HIGH_RATE_THRESHOLD = 4.0        # mg/dL/step rising trend to trigger preemptive correction
TREND_HIGH_BG_MIN = 140.0              # BG must exceed this for trend-based high correction
TREND_LOW_RATE_THRESHOLD = -3.0        # mg/dL/step falling trend to trigger preemptive carb
TREND_LOW_BG_MAX = 100.0               # BG must be below this for trend-based low correction

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

# ============================================================================
# STRESS AND HORMONAL EFFECTS
# ============================================================================

STRESS_PROBABILITY_BASE = 0.05         # Per-day base probability of a stress event
STRESS_LIFESTYLE_WEIGHT = 0.08         # How much lifestyle_consistency reduces stress prob
STRESS_IS_FACTOR_MIN = 1.1             # Minimum IS multiplier during stress (more resistant)
STRESS_IS_FACTOR_MAX = 1.5             # Maximum IS multiplier during stress
STRESS_DURATION_HOURS_MIN = 2.0        # Minimum duration of elevated IS from stress (hours)
STRESS_DURATION_HOURS_MAX = 6.0        # Maximum duration of elevated IS from stress (hours)

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

    # Derived physiological parameters
    is_base: float = 1.0
    icr: float = 10.0
    correction_factor: float = 40.0
    basal_dose: float = 20.0
    hgo_rate: float = 9.0

    # Derived behavioral parameters
    wake_time_hours: float = 8.0
    sleep_duration_hours: float = 7.5
    slow_carb_preference: float = 0.5
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
    is_asleep: bool = True
    is_sick: bool = False
    illness_is_factor: float = 1.0
    last_correction_idx: int = -9999
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
    stress_effects: list = field(default_factory=list)       # (end_idx, is_factor)


# ============================================================================
# CURVE GENERATION UTILITIES
# ============================================================================

def gamma_curve(total_amount: float, k: float, theta: float,
                duration_minutes: float, dt: float = DT_MINUTES) -> np.ndarray:
    """
    Generate a gamma-distributed absorption/action curve.
    Area under curve = total_amount.
    """
    n_steps = int(duration_minutes / dt)
    if n_steps <= 0:
        return np.array([0.0])
    t = np.arange(1, n_steps + 1) * dt  # time in minutes
    # Gamma PDF (unnormalized)
    values = t ** (k - 1) * np.exp(-t / theta)
    # Normalize so the sum of the array equals total_amount (amount per step)
    area = np.sum(values)  # <-- Removed the * dt here
    if area > 0:
        values = values * (total_amount / area)
    return values


def basal_curve(total_amount: float, duration_minutes: float,
                ramp_up_hours: float = 2.0, ramp_down_hours: float = 2.0,
                dt: float = DT_MINUTES) -> np.ndarray:
    """Generate a trapezoidal basal insulin curve."""
    n_steps = int(duration_minutes / dt)
    if n_steps <= 0:
        return np.array([0.0])
    
    ramp_up_steps = int((ramp_up_hours * 60) / dt)
    ramp_down_steps = int((ramp_down_hours * 60) / dt)
    
    curve = np.ones(n_steps)
    if ramp_up_steps > 0:
        curve[:ramp_up_steps] = np.linspace(0, 1, ramp_up_steps)
    if ramp_down_steps > 0:
        curve[-ramp_down_steps:] = np.linspace(1, 0, ramp_down_steps)
        
    # Normalize so the area under the curve equals the total dose
    return curve * (total_amount / np.sum(curve))


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

    # Physiological parameters
    profile.is_base = max(0.3, rng.normal(IS_BASE_MEAN, IS_BASE_SIGMA))
    profile.icr = max(3.0, rng.normal(ICR_MEAN, ICR_SIGMA))
    profile.correction_factor = max(10.0, rng.normal(CORRECTION_FACTOR_MEAN, CORRECTION_FACTOR_SIGMA))

    # Ideal basal balances 24h of HGO: (HGO_BASE_GRAMS_PER_HOUR * 24) / ICR.
    # Competent patients (high s3) stay close to ideal; incompetent ones deviate more.
    ideal_basal = (HGO_BASE_GRAMS_PER_HOUR * 24.0) / profile.icr
    noise_scale = BASAL_DOSE_SIGMA * (1.5 - s3)
    profile.basal_dose = float(np.clip(rng.normal(ideal_basal, noise_scale), 5.0, 40.0))

    profile.hgo_rate = HGO_BASE_GRAMS_PER_HOUR

    # Behavioral parameters derived from skills
    wake_sigma = WAKE_TIME_SIGMA_BASE / (0.3 + 0.7 * s4)
    profile.wake_time_hours = rng.normal(WAKE_TIME_MEAN_HOURS, wake_sigma)
    profile.sleep_duration_hours = rng.normal(SLEEP_DURATION_MEAN_HOURS, SLEEP_DURATION_SIGMA_HOURS)

    profile.slow_carb_preference = SLOW_CARB_PREFERENCE_BASE + SLOW_CARB_PREFERENCE_SKILL_BONUS * s1
    profile.cgm_check_interval_min = (CGM_CHECK_INTERVAL_ATTENTIVE +
                                       (CGM_CHECK_INTERVAL_INATTENTIVE - CGM_CHECK_INTERVAL_ATTENTIVE) * (1 - s2))
    profile.patience_time_min = (PATIENCE_TIME_INCOMPETENT +
                                  (PATIENCE_TIME_COMPETENT - PATIENCE_TIME_INCOMPETENT) * s3)
    profile.carb_count_error_sigma = CARB_COUNT_ERROR_SIGMA_BASE * (1.2 - s3)
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
                BG_CLAMP_MIN, BG_CLAMP_MAX
            )

        self.state.bg_observed = self.state.bg

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
                BG_CLAMP_MIN, BG_CLAMP_MAX
            )

        self.state.bg_observed = self.state.bg

        self._holiday_set = set()
        self._holidays_generated_years = set()
        self._generate_year_holidays(0)
        self._generate_year_holidays(1)

        _init_len = STEPS_PER_DAY * 4
        self._carb_totals = np.zeros(_init_len)
        self._basal_totals = np.zeros(_init_len)
        self._bolus_totals = np.zeros(_init_len)
        self._exercise_totals = np.zeros(_init_len)

        self._pending_events = []
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

        # Daily IS drift
        self._daily_is_drift = self.rng.normal(0, IS_DAILY_DRIFT_SIGMA)
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
        sleep_hours = self.rng.normal(SLEEP_DURATION_MEAN_HOURS, SLEEP_DURATION_SIGMA_HOURS)
        sleep_idx = day_start_idx + int((today_wake + max(12, sleep_hours + 8)) * 60 / DT_MINUTES)

        # Store wake/sleep for the day
        self._today_wake_idx = wake_idx
        self._today_sleep_idx = sleep_idx

        # --- Anomalous event flag for the day ---
        anomalous_today = self.rng.random() < ANOMALOUS_EVENT_PROBABILITY
        anomalous_applied = False  # Only apply to first eligible event

        # --- Basal insulin ---
        basal_time_idx = max(self.state.current_idx, wake_idx + int(self.rng.normal(0, 30) / DT_MINUTES))
        # Slow basal adjustment based on recent BG history (patient learns over days)
        basal_adjustment = 1.0
        if len(self.state.bg_history) > 0:
            recent_bg = self.state.bg_history[-min(len(self.state.bg_history), STEPS_PER_DAY):]
            recent_mean = np.mean(recent_bg)

            if recent_mean > 150:
                overshoot = min((recent_mean - 150) / 100.0, 1.0)
                basal_adjustment = 1.0 + overshoot * (BASAL_CORRECTION_MAX_ADJUSTMENT * eff_s3)
            elif recent_mean < 90:
                undershoot = min((90 - recent_mean) / 50.0, 1.0)
                basal_adjustment = 1.0 - undershoot * (BASAL_CORRECTION_MAX_ADJUSTMENT * eff_s3)

        if self.rng.random() > p.basal_miss_prob:
            # Administer basal
            dose_noise = 1.0 + self.rng.normal(0, BASAL_DOSE_COMPETENCE_NOISE * (1.2 - eff_s3))
            actual_dose = max(1.0, p.basal_dose * dose_noise * basal_adjustment)
            duration = BASAL_DURATION_HOURS * 60
            curve = basal_curve(float(actual_dose), duration, ramp_up_hours=BASAL_RAMP_UP_HOURS, ramp_down_hours=BASAL_RAMP_DOWN_HOURS)
            self._pending_events.append((basal_time_idx, 'basal', {
                'curve': curve, 'label': f'Basal {actual_dose:.1f}U'
            }))

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
            discipline_carb_sigma = MEAL_CARB_SIGMA * (1.0 + 0.5 * (1.0 - eff_s1))
            carb_amount = max(0.0, self.rng.normal(
                carb_mean * discipline_factor * weekend_factor, discipline_carb_sigma))

            # --- Mixed fast/slow carbs (1.3) ---
            # fast_fraction: 0=all slow, 1=all fast.
            # Poor dietary discipline → more fast carbs and more variance.
            slow_pref = SLOW_CARB_PREFERENCE_BASE + SLOW_CARB_PREFERENCE_SKILL_BONUS * eff_s1
            fast_fraction_noise = self.rng.normal(0, 0.15 * (1.2 - eff_s1))
            fast_fraction = float(np.clip((1.0 - slow_pref) + fast_fraction_noise, 0.0, 1.0))

            fast_amount = carb_amount * fast_fraction
            slow_amount = carb_amount * (1.0 - fast_fraction)

            # Apply anomalous event shape modification to one curve this day
            def _maybe_anomalous(k: float, theta: float) -> tuple:
                nonlocal anomalous_applied
                if anomalous_today and not anomalous_applied:
                    anomalous_applied = True
                    k *= float(self.rng.uniform(ANOMALOUS_K_MULT_MIN, ANOMALOUS_K_MULT_MAX))
                    theta *= float(self.rng.uniform(ANOMALOUS_THETA_MULT_MIN, ANOMALOUS_THETA_MULT_MAX))
                return k, theta

            # Fast carb curve
            if fast_amount > 0.5:
                k = FAST_CARB_K * (1 + self.rng.normal(0, CARB_CURVE_K_NOISE))
                theta = FAST_CARB_THETA * (1 + self.rng.normal(0, CARB_CURVE_THETA_NOISE))
                k, theta = _maybe_anomalous(k, theta)
                k = max(1.1, k); theta = max(3.0, theta)
                duration = max(k * theta * 4, 60)
                self._pending_events.append((meal_idx, 'carb', {
                    'curve': gamma_curve(fast_amount, k, theta, duration),
                    'label': f'Meal {fast_amount:.0f}g fast'
                }))

            # Slow carb curve
            if slow_amount > 0.5:
                k = SLOW_CARB_K * (1 + self.rng.normal(0, CARB_CURVE_K_NOISE))
                theta = SLOW_CARB_THETA * (1 + self.rng.normal(0, CARB_CURVE_THETA_NOISE))
                k, theta = _maybe_anomalous(k, theta)
                k = max(1.1, k); theta = max(3.0, theta)
                duration = max(k * theta * 4, 60)
                self._pending_events.append((meal_idx, 'carb', {
                    'curve': gamma_curve(slow_amount, k, theta, duration),
                    'label': f'Meal {slow_amount:.0f}g slow'
                }))

            # Protein/fat slow curve (always present)
            pf_curve = gamma_curve(PROTEIN_FAT_EQUIV_GRAMS, PROTEIN_FAT_GAMMA_K,
                                   PROTEIN_FAT_GAMMA_THETA,
                                   PROTEIN_FAT_GAMMA_K * PROTEIN_FAT_GAMMA_THETA * 4)
            self._pending_events.append((meal_idx, 'carb', {
                'curve': pf_curve, 'label': f'Protein/fat {PROTEIN_FAT_EQUIV_GRAMS:.0f}g equiv'
            }))

            # --- Bolus for this meal ---
            carb_estimate = max(0, carb_amount * (1 + self.rng.normal(0, p.carb_count_error_sigma)))

            bolus_skip_prob = 0.0
            if i >= MEALS_BASE:
                bolus_skip_prob = 0.3 * (1 - eff_s3)

            if self.rng.random() > bolus_skip_prob and carb_estimate > 0:
                bolus_dose = carb_estimate / p.icr
                bolus_timing_offset = self.rng.normal(p.bolus_timing_mean, p.bolus_timing_sigma)
                bolus_idx = max(self.state.current_idx, meal_idx + int(bolus_timing_offset / DT_MINUTES))

                bolus_k = BOLUS_GAMMA_K * (1 + self.rng.normal(0, 0.05))
                bolus_theta = BOLUS_GAMMA_THETA * (1 + self.rng.normal(0, 0.05))
                bolus_duration = BOLUS_DURATION_HOURS * 60
                bolus_curve = gamma_curve(bolus_dose, max(1.5, bolus_k),
                                          max(5.0, bolus_theta), bolus_duration)
                self._pending_events.append((bolus_idx, 'bolus', {
                    'curve': bolus_curve, 'label': f'Bolus {bolus_dose:.1f}U'
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
            ex_duration = max(10.0, self.rng.normal(EXERCISE_DURATION_MEAN_MIN, EXERCISE_DURATION_SIGMA_MIN))
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
            s.stress_effects.append((end_idx, is_factor))

        # Sort events by time
        self._pending_events.sort(key=lambda x: x[0])

    def _compute_insulin_resistance(self, time_idx: int) -> float:
        """Compute insulin resistance factor at a given time index, including diurnal pattern, drift, illness, noise."""
        s = self.state

        # Time of day in hours
        hour = (time_idx * DT_MINUTES / 60.0) % 24.0

        # Multi-peak diurnal pattern
        morning = IS_MORNING_AMPLITUDE * np.exp(-0.5 * ((hour - IS_MORNING_PEAK_HOUR - self._daily_is_phase_shift) / 2.0) ** 2)
        evening = IS_EVENING_AMPLITUDE * np.exp(-0.5 * ((hour - IS_EVENING_PEAK_HOUR) / 2.5) ** 2)
        night_hour = hour if hour < 12 else hour - 24
        night = -IS_NIGHT_DIP_AMPLITUDE * np.exp(-0.5 * ((night_hour - IS_NIGHT_DIP_HOUR) / 2.0) ** 2)
        diurnal = 1.0 + morning + evening + night

        is_val = self.patient.is_base * diurnal * (1.0 + self._daily_is_drift)

        # Illness
        if s.is_sick:
            is_val *= s.illness_is_factor

        # Post-exercise IS reduction (aerobic exercise increases insulin sensitivity for hours)
        exercise_reduction = 0.0
        active_ex_effects = []
        for (start_idx, end_idx, reduction) in s.exercise_is_effects:
            if time_idx < end_idx:
                active_ex_effects.append((start_idx, end_idx, reduction))
                if time_idx >= start_idx:
                    exercise_reduction += reduction
        s.exercise_is_effects = active_ex_effects
        if exercise_reduction > 0:
            is_val *= (1.0 - min(0.30, exercise_reduction))

        # Stress IS effect (transient insulin resistance from cortisol/adrenaline)
        stress_factor = 1.0
        active_stress = []
        for (end_idx, factor) in s.stress_effects:
            if time_idx < end_idx:
                active_stress.append((end_idx, factor))
                stress_factor = max(stress_factor, factor)
        s.stress_effects = active_stress
        if stress_factor > 1.0:
            is_val *= stress_factor

        # Fast noise
        is_val *= (1.0 + self.rng.normal(0, IS_FAST_NOISE_SIGMA))

        return max(0.2, is_val)

    def _compute_cgm_observation(self, true_bg: float) -> float:
        """Compute CGM reading with lag and proportional noise."""
        noise_sigma = CGM_NOISE_FRACTION * true_bg
        observed = true_bg + self.rng.normal(0, noise_sigma)
        return np.clip(observed, BG_CLAMP_MIN, BG_CLAMP_MAX)

    def _check_and_correct(self, time_idx: int):
        """Patient checks CGM and possibly corrects highs/lows."""
        p = self.patient
        s = self.state

        is_awake = self._today_wake_idx <= time_idx < self._today_sleep_idx
        if not is_awake:
            if s.bg_observed < 55 or s.bg_observed > 350:
                delay_steps = int(self.rng.exponential(HYPO_DETECTION_ASLEEP_LAMBDA) / DT_MINUTES)
                if delay_steps > 0:
                    return
            else:
                return

        # Check interval
        steps_since_check = time_idx - s.last_cgm_check_idx
        check_interval_steps = int(p.cgm_check_interval_min / DT_MINUTES)
        if steps_since_check < check_interval_steps:
            return

        s.last_cgm_check_idx = time_idx

        # Compute insulin on board (IOB) from the pre-accumulated insulin array.
        # This is O(n_future) with numpy, faster than iterating active_curves.
        if time_idx < len(self._bolus_totals):
            iob = float(np.sum(self._bolus_totals[time_idx:]))
        else:
            iob = 0.0

        # --- Handle hypoglycemia ---
        if s.bg_observed < BG_LOW_THRESHOLD:
            severity = max(0, BG_LOW_THRESHOLD - s.bg_observed)
            correction_grams = HYPO_CORRECTION_BASE_GRAMS + p.panic_factor * severity / 20.0

            if s.bg_observed < RAGE_EAT_BG_THRESHOLD:
                rage_prob = RAGE_EAT_PROBABILITY_BASE * (1.2 - p.dosing_competence)
                if self.rng.random() < rage_prob:
                    correction_grams = self.rng.uniform(RAGE_EAT_CARB_MIN, RAGE_EAT_CARB_MAX)

            k = FAST_CARB_K
            theta = FAST_CARB_THETA
            duration = max(k * theta * 4, 60)
            curve = gamma_curve(correction_grams, k, theta, duration)
            self.inject_curve(curve, time_idx, 'correction_carb',
                              f'Hypo correction {correction_grams:.0f}g')

        # --- Handle hyperglycemia ---
        elif s.bg_observed > BG_HIGH_THRESHOLD:
            steps_since_correction = time_idx - s.last_correction_idx
            urgency = max(1.0, (s.bg_observed - 250) / 50.0) if s.bg_observed > 250 else 1.0
            patience_steps = int(p.patience_time_min / (DT_MINUTES * urgency))

            if steps_since_correction >= patience_steps:
                # IOB-aware correction: skilled patients subtract remaining active insulin
                iob_equiv_bg_drop = iob * p.correction_factor
                iob_consideration = iob_equiv_bg_drop * p.dosing_competence
                adjusted_excess = max(0.0, (s.bg_observed - BG_TARGET) - iob_consideration)
                correction_dose = adjusted_excess / p.correction_factor
                correction_dose *= (1 + self.rng.normal(0, p.carb_count_error_sigma * 0.5))
                correction_dose = max(0.5, correction_dose)

                if s.bg_observed > RAGE_BOLUS_BG_THRESHOLD:
                    rage_prob = RAGE_BOLUS_PROBABILITY_BASE * (1.2 - p.dosing_competence)
                    if self.rng.random() < rage_prob:
                        rage_mult = self.rng.uniform(RAGE_BOLUS_MULTIPLIER_MIN, RAGE_BOLUS_MULTIPLIER_MAX)
                        correction_dose *= rage_mult

                bolus_curve = gamma_curve(correction_dose, BOLUS_GAMMA_K,
                                          BOLUS_GAMMA_THETA, BOLUS_DURATION_HOURS * 60)
                self.inject_curve(bolus_curve, time_idx, 'bolus',
                                  f'Correction {correction_dose:.1f}U')
                s.last_correction_idx = time_idx

        # --- Trend-based anticipatory corrections ---
        elif len(s.bg_history) >= TREND_CORRECTION_WINDOW_STEPS:
            steps_since_correction = time_idx - s.last_correction_idx
            patience_steps = int(p.patience_time_min / DT_MINUTES)
            if steps_since_correction >= patience_steps:
                window = s.bg_history[-TREND_CORRECTION_WINDOW_STEPS:]
                trend = (window[-1] - window[0]) / (TREND_CORRECTION_WINDOW_STEPS - 1)

                if (trend > TREND_HIGH_RATE_THRESHOLD and
                        s.bg_observed > TREND_HIGH_BG_MIN and
                        s.bg_observed <= BG_HIGH_THRESHOLD):
                    if self.rng.random() < p.attentiveness:
                        projected_rise = trend * TREND_CORRECTION_WINDOW_STEPS * 2
                        correction_dose = max(0.5, projected_rise * p.attentiveness / p.correction_factor)
                        bolus_curve = gamma_curve(correction_dose, BOLUS_GAMMA_K,
                                                  BOLUS_GAMMA_THETA, BOLUS_DURATION_HOURS * 60)
                        self.inject_curve(bolus_curve, time_idx, 'bolus',
                                          f'Trend corr {correction_dose:.1f}U')
                        s.last_correction_idx = time_idx

                elif (trend < TREND_LOW_RATE_THRESHOLD and
                          s.bg_observed < TREND_LOW_BG_MAX and
                          s.bg_observed >= BG_LOW_THRESHOLD):
                    if self.rng.random() < p.attentiveness:
                        correction_grams = float(np.clip(
                            abs(trend) * TREND_CORRECTION_WINDOW_STEPS * 2.0, 5.0, 20.0))
                        k = FAST_CARB_K
                        theta = FAST_CARB_THETA
                        duration = max(k * theta * 4, 60)
                        curve = gamma_curve(correction_grams, k, theta, duration)
                        self.inject_curve(curve, time_idx, 'correction_carb',
                                          f'Trend corr {correction_grams:.0f}g')

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
            # Scatter-add into accumulation arrays (O(len(curve)), done once per event)
            self._add_to_totals(curve, event_time, event_type)
            # Keep active_curves populated for external consumers
            s.active_curves.append(ActiveCurve(
                start_time_idx=event_time,
                values=curve,
                curve_type=event_type,
                label=event_data.get('label', '')
            ))
            # Schedule post-exercise IS sensitivity boost
            if event_type == 'exercise':
                ex_dur = event_data.get('duration_min', EXERCISE_DURATION_MEAN_MIN)
                effect_start = event_time + int(ex_dur / DT_MINUTES)
                effect_end = effect_start + int(EXERCISE_IS_DURATION_HOURS * 60 / DT_MINUTES)
                reduction = min(0.30, EXERCISE_IS_REDUCTION * (ex_dur / EXERCISE_DURATION_MEAN_MIN))
                s.exercise_is_effects.append((effect_start, effect_end, reduction))

        # --- Compute HGO for this step ---
        hgo_rate = p.hgo_rate * (1 + self.rng.normal(0, HGO_NOISE_SIGMA))
        hgo_value = hgo_rate * (DT_MINUTES / 60.0)

        # Apply alcohol HGO suppression
        alcohol_hgo_factor = 1.0
        active_alcohol = []
        for (start_idx, end_idx, hgo_factor) in s.alcohol_effects:
            if idx < end_idx:
                active_alcohol.append((start_idx, end_idx, hgo_factor))
                if idx >= start_idx:
                    alcohol_hgo_factor = min(alcohol_hgo_factor, hgo_factor)
        s.alcohol_effects = active_alcohol
        hgo_value *= alcohol_hgo_factor

        # --- Read per-step contributions from pre-computed accumulation arrays (O(1)) ---
        total_carb = float(self._carb_totals[idx]) if idx < len(self._carb_totals) else 0.0
        total_insulin = (float(self._basal_totals[idx]) + float(self._bolus_totals[idx])) if idx < len(self._basal_totals) else 0.0
        total_exercise = float(self._exercise_totals[idx]) if idx < len(self._exercise_totals) else 0.0

        # Remove expired entries from active_curves (memory management for external consumers)
        s.active_curves = [c for c in s.active_curves
                           if (idx - c.start_time_idx) < len(c.values)]

        # --- Insulin sensitivity ---
        insulin_resistance_factor = self._compute_insulin_resistance(idx)

        # --- Compute BG delta ---
        insulin_carb_equiv = total_insulin * p.icr
        effective_carb_load = (total_carb + hgo_value - total_exercise) * insulin_resistance_factor
        bg_delta = BG_SCALE_FACTOR * (effective_carb_load - insulin_carb_equiv)

        # Physiological guardrails
        if s.bg > RENAL_THRESHOLD:
            bg_delta -= (s.bg - RENAL_THRESHOLD) * RENAL_CLEARANCE_RATE

        if s.bg < COUNTER_REGULATORY_THRESHOLD:
            bg_delta += COUNTER_REGULATORY_RATE * (COUNTER_REGULATORY_THRESHOLD - s.bg) / COUNTER_REGULATORY_THRESHOLD

        # Update BG
        s.bg = np.clip(s.bg + bg_delta, BG_CLAMP_MIN, BG_CLAMP_MAX)

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
            'total_exercise': total_exercise,
            'insulin_resistance': insulin_resistance_factor,
            'hgo': hgo_value,
            'effective_carb_load': effective_carb_load,
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
            'total_carb': [], 'total_insulin': [], 'total_exercise': [],
            'insulin_resistance': [], 'hgo': [], 'effective_carb_load': [],
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
