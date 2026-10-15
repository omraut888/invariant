"""E4's signal S_t = log h_{t+1} must be computable from returns up to t only."""

import numpy as np
import pytest
from tests.fixtures.vol_null import VolNullData, vol_null_config

pytestmark = pytest.mark.leakage


def test_signal_is_a_function_of_past_returns() -> None:
    cfg = vol_null_config()
    data, _ = VolNullData(cfg)(3, 2000)
    s, r = data[:, 0], data[:, 1]
    g = cfg.garch
    a = r - cfg.mu  # global null: R_t = mu + a_t exactly
    # Recurse h_{t+1} = omega + alpha a_t^2 + beta h_t from the first value only. Each step reads
    # a_t, never a later return.
    h = np.empty(len(r) + 1)
    h[0] = np.nan
    h[1] = np.exp(s[0])
    for t in range(1, len(r)):
        h[t + 1] = g.omega + g.alpha * a[t] ** 2 + g.beta * h[t]
    np.testing.assert_allclose(np.log(h[2:]), s[1:], rtol=1e-10)


def test_realized_vol_signal_reads_only_past_returns() -> None:
    from tests.fixtures.vol_null import RV_WINDOW

    data, _ = VolNullData(vol_null_config(), signal="rv22")(4, 600)
    s, r = data[:, 0], data[:, 1]
    # For t >= 21 the window lies inside the returned rows: S_t = log mean(r_{t-21..t}^2).
    for t in range(RV_WINDOW - 1, 600, 37):
        assert np.log(np.mean(r[t - RV_WINDOW + 1 : t + 1] ** 2)) == pytest.approx(s[t], rel=1e-12)
