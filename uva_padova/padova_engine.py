"""Adapter that drives the authoritative UVA/Padova 2008 ODE model from a
discrete carb/insulin event stream.

The reference engine is the FDA-accepted UVA/Padova 2008 in-silico model as
implemented by ``simglucose`` (Jinyu Xie). We use only its ODE core
(``T1DPatient``) and the 30 bundled virtual-patient parameter sets — not its
reinforcement-learning environment.

``simglucose``'s package ``__init__`` and ``t1dpatient.py`` import ``gym`` and
``pkg_resources`` at load time. ``gym`` is needed solely by the RL wrapper, and
``pkg_resources`` was dropped by ``setuptools>=81``. Both are stubbed into
``sys.modules`` below so the pure ODE engine loads with numpy/scipy/pandas
only. Install the reference engine without its RL extras:

    pip install --no-deps simglucose>=0.2.11
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types
from typing import Optional

import numpy as np


def _install_shims() -> None:
    """Stub the RL/packaging imports that simglucose touches at load time."""
    if "gym.envs.registration" not in sys.modules:
        gym = types.ModuleType("gym")
        envs = types.ModuleType("gym.envs")
        reg = types.ModuleType("gym.envs.registration")
        reg.register = lambda *a, **k: None  # type: ignore[attr-defined]
        gym.envs = envs  # type: ignore[attr-defined]
        envs.registration = reg  # type: ignore[attr-defined]
        sys.modules.update({"gym": gym, "gym.envs": envs, "gym.envs.registration": reg})
    def _resource_filename(package: str, resource: str) -> str:
        spec = importlib.util.find_spec(package)
        assert spec is not None and spec.origin is not None
        return os.path.join(os.path.dirname(spec.origin), resource)

    # setuptools>=81 dropped pkg_resources; some builds still ship a partial
    # stub without resource_filename. Handle both: absent -> full stub;
    # present-but-incomplete -> patch in the one function simglucose needs.
    try:
        import pkg_resources as _pr  # type: ignore
        if not hasattr(_pr, "resource_filename"):
            _pr.resource_filename = _resource_filename  # type: ignore[attr-defined]
    except Exception:
        pr = types.ModuleType("pkg_resources")
        pr.resource_filename = _resource_filename  # type: ignore[attr-defined]
        sys.modules["pkg_resources"] = pr


_install_shims()

import pandas as pd  # noqa: E402
import simglucose  # noqa: E402
from simglucose.patient.t1dpatient import Action, T1DPatient  # noqa: E402

_PARAMS_DIR = os.path.join(os.path.dirname(simglucose.__file__), "params")
_PARAM_CSV = os.path.join(_PARAMS_DIR, "vpatient_params.csv")
_QUEST_CSV = os.path.join(_PARAMS_DIR, "Quest.csv")
_QUEST = pd.read_csv(_QUEST_CSV).set_index("Name")


def patient_names(group: Optional[str] = None) -> list:
    """All 30 virtual-patient names, optionally filtered to a cohort.

    Groups: ``adult``, ``adolescent``, ``child`` (10 each).
    """
    names = pd.read_csv(_PARAM_CSV)["Name"].tolist()
    if group is not None:
        names = [n for n in names if n.startswith(group)]
    return names


class PadovaPatient:
    """Thin wrapper around ``T1DPatient`` exposing a discrete-event replay driver."""

    SAMPLE_MIN = 1  # the UVA/Padova ODE is integrated at 1-minute resolution

    def __init__(self, name: str):
        self.name = name
        self._patient = T1DPatient.withName(name)

    @property
    def body_weight_kg(self) -> float:
        return float(self._patient._params.BW)

    @property
    def basal_rate_u_per_min(self) -> float:
        """Steady-state (u2ss) basal infusion rate that balances EGP, in U/min."""
        p = self._patient._params
        return float(p.u2ss * p.BW / 6000.0)

    @property
    def init_gsub(self) -> float:
        """Subcutaneous glucose (mg/dL) at the model's default initial state."""
        return float(self._patient.observation.Gsub)

    @property
    def cr(self) -> float:
        """Carb ratio (g per unit) from the bundled Quest dosing table."""
        return float(_QUEST.loc[self.name, "CR"])

    @property
    def cf(self) -> float:
        """Correction factor (mg/dL per unit) from the bundled Quest table."""
        return float(_QUEST.loc[self.name, "CF"])

    def replay_self_dosed(self, meals, n_minutes: int,
                          target_bg: float = 130.0, max_correction_u: float = 6.0):
        """Integrate the ODE driven only by a shared MEAL schedule, with insulin
        dosed for *this* patient's own physiology.

        ``meals`` is a list of ``(minute, grams)``. Each meal is covered by a
        bolus of ``grams / CR`` plus a bounded correction ``(Gsub - target)/CF``
        (only when above target), and the patient runs on its steady-state
        (u2ss) basal throughout. This is the textbook basal-bolus dosing the
        UVA/Padova virtual patients are constructed around, so it keeps them in
        range and makes their meal excursions comparable in amplitude.

        Returns subcutaneous glucose ``Gsub`` (mg/dL) at every minute.
        """
        p = self._patient
        p.reset()
        basal = self.basal_rate_u_per_min
        cr, cf = self.cr, self.cf
        meal_g = {int(m): 0.0 for m, _ in meals}
        for m, g in meals:
            meal_g[int(m)] += float(g)
        gsub = np.empty(n_minutes)
        for t in range(n_minutes):
            cho = meal_g.get(t, 0.0)
            ins = basal
            if cho > 0.0:
                bolus = cho / cr
                correction = max(0.0, (float(p.observation.Gsub) - target_bg) / cf)
                bolus += min(correction, max_correction_u)
                ins += bolus
            p.step(Action(CHO=cho, insulin=ins))
            gsub[t] = p.observation.Gsub
        return gsub

    def replay(self, carb_g_per_min: np.ndarray, insulin_u_per_min: np.ndarray) -> np.ndarray:
        """Integrate the ODE minute-by-minute under the supplied inputs.

        Parameters are aligned 1-minute arrays of identical length:
          * ``carb_g_per_min[t]``    -- grams of CHO ingested at minute ``t`` as an
            impulse; the model self-paces digestion at ``EAT_RATE`` (5 g/min).
          * ``insulin_u_per_min[t]`` -- total insulin delivered at minute ``t`` in
            U/min: the continuous basal rate plus any bolus impulse for that minute.

        Returns subcutaneous glucose ``Gsub`` (mg/dL) sampled at every minute.
        """
        p = self._patient
        p.reset()
        n = len(carb_g_per_min)
        gsub = np.empty(n)
        for t in range(n):
            p.step(Action(CHO=float(carb_g_per_min[t]), insulin=float(insulin_u_per_min[t])))
            gsub[t] = p.observation.Gsub
        return gsub
