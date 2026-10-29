import numpy as np
import pytest

from invariant.stages.adversarial.state import trailing_volatility, volatility_state


def test_trailing_volatility_matches_a_direct_computation() -> None:
    x = np.random.default_rng(3).standard_normal(200)
    v = trailing_volatility(x, 21)
    assert np.isnan(v[:20]).all()
    for t in (20, 57, 199):
        assert v[t] == pytest.approx(x[t - 20 : t + 1].std(ddof=1), rel=1e-12)


def test_state_share_is_one_minus_q_and_uses_only_past_returns() -> None:
    gen = np.random.default_rng(4)
    x = gen.standard_normal(3000) * np.where(np.arange(3000) % 400 < 100, 3.0, 1.0)
    dates = np.arange(1260, 2999, dtype=np.int64)
    state, threshold = volatility_state(x, dates, quantile=0.7)
    assert state.mean() == pytest.approx(0.3, abs=1 / len(dates))
    # The label of row k (P&L on dates[k] + 1) must not depend on that day's return.
    y = x.copy()
    y[dates[100] + 1] = 1e6
    changed, _ = volatility_state(y, dates, quantile=0.7)
    assert changed[100] == state[100]
    assert threshold > 0


def test_state_rejects_bad_inputs() -> None:
    x = np.ones(100)
    with pytest.raises(ValueError, match="quantile"):
        volatility_state(x, np.arange(30, 99), quantile=1.0)
    with pytest.raises(ValueError, match="trailing"):
        volatility_state(x, np.arange(5, 99))
