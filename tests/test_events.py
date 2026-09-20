import numpy as np
import pytest

from simulator import (EVENT_CHANNELS, EventLog, T1DMSimulator, bateman_peak_min,
                       gamma_curve, gamma_peak_min)


def test_event_log_sums_doses_and_weights_descriptors():
    log = EventLog()
    log.add(7, 'bolus', 1.0, peak_min=50.0, dur_h=4.0)
    log.add(7, 'insulin', 3.0, peak_min=70.0, dur_h=6.0)
    log.add(7, 'correction_carb', 15.0, gi=90.0)
    step = log.step(7)
    assert tuple(step) == EVENT_CHANNELS
    assert step['bolus_u'] == 4.0 and step['bolus_peak_min'] == 65.0 and step['bolus_dur_h'] == 5.5
    assert step['carb_g'] == 15.0 and step['carb_gi'] == 90.0
    assert step['basal_u'] == 0.0 and step['basal_peak_min'] == 0.0
    assert not any(log.step(8).values())


def test_event_log_ignores_an_empty_dose_and_rejects_a_wrong_descriptor():
    log = EventLog()
    log.add(0, 'carb', 0.0, gi=50.0)
    assert not any(log.step(0).values())
    with pytest.raises(AssertionError):
        log.add(0, 'carb', 10.0, peak_min=60.0)


def test_peak_helpers():
    assert gamma_peak_min(3.0, 25.0) == 50.0
    t = np.arange(0.0, 48.0, 1e-3)
    ka, ke = 0.477, 0.0499
    curve = np.exp(-ke * t) - np.exp(-ka * t)
    assert bateman_peak_min(ka, ke) == pytest.approx(60.0 * t[curve.argmax()], abs=0.1)


def test_injected_event_lands_on_its_step():
    sim = T1DMSimulator(seed=3)
    idx = sim.state.current_idx
    sim.inject_curve(gamma_curve(30.0, 3.0, 20.0, 240.0), idx, 'carb',
                     event={'dose': 30.0, 'gi': 70.0})
    step = sim.generate()
    assert step['carb_g'] >= 30.0 and step['carb_gi'] > 0.0


def test_event_totals_track_the_curve_totals():
    data = T1DMSimulator(seed=11).generate_hours(24 * 12)
    n = len(data['bg'])
    # Curves still running at the end have delivered part of their dose: compare the early 10 days.
    cut = n - 2 * 288
    carb_ev, carb_cv = data['carb_g'][:cut].sum(), data['total_carb'][:cut].sum()
    assert carb_ev > 0 and carb_cv == pytest.approx(carb_ev, rel=0.1)
    assert data['bolus_u'].sum() > 0 and data['basal_u'].sum() > 0
    for dose, descriptors in (('bolus_u', ('bolus_peak_min', 'bolus_dur_h')),
                              ('basal_u', ('basal_peak_min', 'basal_dur_h'))):
        for d in descriptors:
            assert np.array_equal(data[d] > 0, data[dose] > 0)
    # A guessed GI clips at 0, so carb_gi is positive on every carb step but the clipped ones.
    carb, gi = data['carb_g'] > 0, data['carb_gi'] > 0
    assert not (gi & ~carb).any()
    assert (carb & ~gi).sum() <= 1
