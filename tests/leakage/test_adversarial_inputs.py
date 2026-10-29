"""Stage 4's volatility state is fixed before the P&L it labels (docs/adr/0016).

The P&L of row k is earned on ``dates[k] + 1``. Its state label must depend only on returns up to
``dates[k]``, so a state can never be selected on its own outcome (which would bias the per-state
P&L moments Stage 4 decides on). The threshold is a full-sample quantile by design (Stage 4 is an
ex-post certification of the out-of-sample path), so the property is checked with the threshold
held fixed: perturbing returns after ``dates[k]`` cannot change ``v[dates[k]]``.
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from invariant.stages.adversarial.state import trailing_volatility

pytestmark = pytest.mark.leakage


@settings(max_examples=60, deadline=None)
@given(t=st.integers(20, 498), seed=st.integers(0, 2**32 - 1))
def test_trailing_volatility_at_t_ignores_returns_after_t(t: int, seed: int) -> None:
    gen = np.random.default_rng(seed)
    x = gen.standard_normal(500)
    y = x.copy()
    y[t + 1 :] = gen.standard_normal(500 - t - 1) * 10.0
    np.testing.assert_array_equal(trailing_volatility(x)[: t + 1], trailing_volatility(y)[: t + 1])
