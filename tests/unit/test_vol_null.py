"""The E4 DGP must be a null in the mean and a real stress test for the variance."""

import numpy as np
import pytest
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


def test_variants_are_valid_nulls_with_distinct_persistence() -> None:
    from tests.fixtures.vol_null import vol_null_config, vol_null_variants

    names = [s.name for s in vol_null_variants(5)]
    assert names == ["e4_vol_null", "e4_low_persist", "e4_high_persist", "e4_rv22"]
    lo, hi = vol_null_config(0.10, 0.80).garch, vol_null_config(0.04, 0.95).garch
    assert lo.alpha + lo.beta == pytest.approx(0.90) and hi.alpha + hi.beta == pytest.approx(0.99)
    for sc in vol_null_variants(5):
        data, _ = sc.build(2, 3000)
        assert data.shape == (3000, 2) and np.isfinite(data).all()
        s_lag, r = data[:-1, 0], data[1:, 1]
        assert abs(np.corrcoef(s_lag, r)[0, 1]) < 4 / np.sqrt(len(r))
        assert np.corrcoef(s_lag, r**2)[0, 1] > 0.05, sc.name


def test_stambaugh_dgp() -> None:
    from tests.fixtures.stambaugh import SIGNAL as S
    from tests.fixtures.stambaugh import StambaughData, stambaugh_scenario

    data, names = StambaughData()(9, 20000)
    assert names == (S, TARGET)
    x, r = data[:, 0], data[:, 1]
    u = x[1:] - 0.99 * x[:-1]
    assert np.corrcoef(u, r[1:])[0, 1] == pytest.approx(-0.8, abs=0.02)
    rho_hat = np.polyfit(x[:-1], x[1:], 1)[0]
    assert rho_hat == pytest.approx(0.99, abs=0.005)
    assert abs(np.corrcoef(x[:-1], r[1:])[0, 1]) < 4 / np.sqrt(len(r))  # null in the mean
    assert StambaughData()(9, 500)[0].tolist() == StambaughData()(9, 500)[0].tolist()
    gates = [f for f in stambaugh_scenario(5).families if f.role is FamilyRole.GATING_NULL]
    assert [g.name for g in gates] == [f"{S}->R"]
