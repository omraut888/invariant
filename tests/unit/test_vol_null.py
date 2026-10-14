"""The E4 DGP must be a null in the mean and a real stress test for the variance."""

import numpy as np
from tests.fixtures.vol_null import SIGNAL, VolNullData, vol_null_config, vol_null_scenario

from invariant.stages.causal.calibration import TARGET, FamilyRole


def test_shape_and_names() -> None:
    data, names = VolNullData(vol_null_config())(5, 1000)
    assert data.shape == (1000, 2) and names == (SIGNAL, TARGET)
    assert np.isfinite(data).all()


def test_null_in_mean_but_signal_predicts_variance() -> None:
    data, _ = VolNullData(vol_null_config())(11, 6000)
    s_lag, r = data[:-1, 0], data[1:, 1]
    t = len(r)
    # Null: no linear relation between S_{t-1} and R_t (within 4 standard errors).
    assert abs(np.corrcoef(s_lag, r)[0, 1]) < 4 / np.sqrt(t)
    # Stress: S_{t-1} forecasts R_t^2, so S^2 is correlated with the conditional variance.
    assert np.corrcoef(s_lag, r**2)[0, 1] > 0.1


def test_scenario_has_one_gate() -> None:
    sc = vol_null_scenario(5)
    gates = [f for f in sc.families if f.role is FamilyRole.GATING_NULL]
    assert [g.name for g in gates] == [f"{SIGNAL}->R"] and gates[0].lags == (1, 2, 3, 4, 5)
