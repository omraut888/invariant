"""Stage 3 look-ahead guard: every arm's weight at date t uses only data dated <= t.

Perturbing features and returns after a cut date t* leaves every weight decided at dates <= t*
unchanged, and every realised return ``r[d + 1 + e]`` that lies at or before t*. That covers
refits, walk-forward CV and the forecast-confidence radius, at execution lags e = 0, 1 and 2.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from invariant.core.run_hash import RngFactory
from invariant.stages.dro.walkforward import Arm, WalkForwardConfig, run_walk_forward

pytestmark = pytest.mark.leakage

CFG = WalkForwardConfig(
    c=0.03,
    window=120,
    refit_every=7,
    cv_every=30,
    cv_valid=40,
    ridge_grid=(0.0, 0.3),
    eps_grid=(0.0, 0.05),
    fc_draws=500,
)


@settings(max_examples=6, deadline=None)
@given(st.integers(0, 2**32 - 1), st.integers(130, 230), st.integers(0, 2))
def test_no_arm_uses_data_after_t(seed: int, cut: int, lag: int) -> None:
    cfg = replace(CFG, lag=lag)
    rng = np.random.default_rng(seed)
    t, n = 260, 3
    x = rng.standard_normal((t, 2))
    r = 0.01 * rng.standard_normal((t, n)) + 0.002 * np.r_[np.zeros((1, 2)), x[:-1]] @ np.ones(
        (2, n)
    )
    base = run_walk_forward(x, r, list(Arm), cfg, RngFactory(1))
    x2, r2 = x.copy(), r.copy()
    x2[cut + 1 :] = rng.standard_normal(x2[cut + 1 :].shape) * 3
    r2[cut + 1 :] = rng.standard_normal(r2[cut + 1 :].shape) * 0.05
    other = run_walk_forward(x2, r2, list(Arm), cfg, RngFactory(1))
    for arm in Arm:
        dates = base[arm].dates
        known = dates <= cut
        np.testing.assert_array_equal(base[arm].weights[known], other[arm].weights[known])
        realised = dates + 1 + lag <= cut
        np.testing.assert_array_equal(base[arm].returns[realised], other[arm].returns[realised])
