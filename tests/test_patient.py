"""
Tests for patient generation.

Verifies that:
- Skills are within expected [SKILL_MIN, SKILL_MAX] range
- Derived parameters are physiologically plausible
- Basal dose is tied to HGO and ICR (not independently sampled)
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulator import (
    generate_patient, SKILL_MIN, SKILL_MAX, HGO_BASE_GRAMS_PER_HOUR,
    BASAL_DOSE_SIGMA
)


def make_patient(seed: int = 0):
    rng = np.random.default_rng(seed)
    return generate_patient(rng)


class TestSkillRange:
    def test_skills_within_bounds(self):
        """All four skills stay within [SKILL_MIN, SKILL_MAX]."""
        for seed in range(50):
            p = make_patient(seed)
            for attr in ['dietary_discipline', 'attentiveness',
                         'dosing_competence', 'lifestyle_consistency']:
                val = getattr(p, attr)
                assert SKILL_MIN <= val <= SKILL_MAX, (
                    f"seed={seed}: {attr}={val:.3f} outside [{SKILL_MIN}, {SKILL_MAX}]")

    def test_skills_span_reasonable_range(self):
        """With many seeds, we see both low and high skill patients."""
        skills = [make_patient(s).dietary_discipline for s in range(200)]
        assert min(skills) < 0.63, "Never generated low-skill patients"
        assert max(skills) > 0.70, "Never generated high-skill patients"


class TestPhysiologicalParameters:
    def test_icr_positive(self):
        """ICR (insulin-to-carb ratio) is always positive."""
        for seed in range(30):
            p = make_patient(seed)
            assert p.icr > 0, f"seed={seed}: ICR={p.icr} is not positive"

    def test_correction_factor_positive(self):
        for seed in range(30):
            p = make_patient(seed)
            assert p.correction_factor > 0

    def test_is_base_positive(self):
        for seed in range(30):
            p = make_patient(seed)
            assert p.is_base > 0

    def test_basal_dose_tied_to_hgo_icr(self):
        """Basal dose should be near ideal = (HGO_BASE_GRAMS_PER_HOUR * 24) / ICR.

        The ideal dose balances 24h of hepatic glucose output. Patients should be
        within a few sigma of the ideal (not at an independent arbitrary value).
        """
        for seed in range(50):
            p = make_patient(seed)
            ideal = (HGO_BASE_GRAMS_PER_HOUR * 24.0) / p.icr
            # Allow generous tolerance (skilled patients are close, unskilled deviate)
            tolerance = BASAL_DOSE_SIGMA * 5.0
            assert abs(p.basal_dose - ideal) < tolerance, (
                f"seed={seed}: basal_dose={p.basal_dose:.1f} too far from "
                f"ideal={ideal:.1f} (tolerance={tolerance:.1f})")

    def test_basal_dose_clamped(self):
        """Basal dose is always within clinically plausible range [5, 40] U."""
        for seed in range(100):
            p = make_patient(seed)
            assert 5.0 <= p.basal_dose <= 40.0, (
                f"seed={seed}: basal_dose={p.basal_dose:.1f} outside [5, 40]")


class TestBehavioralParameters:
    def test_high_skill_patients_have_shorter_cgm_interval(self):
        """More attentive patients check their CGM more often."""
        low_skill_intervals = []
        high_skill_intervals = []
        for seed in range(200):
            p = make_patient(seed)
            if p.attentiveness < 0.65:
                low_skill_intervals.append(p.cgm_check_interval_min)
            elif p.attentiveness > 0.80:
                high_skill_intervals.append(p.cgm_check_interval_min)

        if low_skill_intervals and high_skill_intervals:
            assert np.mean(low_skill_intervals) > np.mean(high_skill_intervals), (
                "Less attentive patients should check CGM less frequently")

    def test_exercise_probability_positive(self):
        for seed in range(30):
            p = make_patient(seed)
            assert 0.0 <= p.exercise_probability <= 1.0

    def test_slow_carb_preference_skill_relationship(self):
        """More dietary discipline → higher slow carb preference."""
        low_disc = [make_patient(s) for s in range(200) if make_patient(s).dietary_discipline < 0.65]
        high_disc = [make_patient(s) for s in range(200) if make_patient(s).dietary_discipline > 0.80]
        if low_disc and high_disc:
            assert np.mean([p.slow_carb_preference for p in low_disc]) < \
                   np.mean([p.slow_carb_preference for p in high_disc])
