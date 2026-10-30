"""The reported-only stress diagnostic attached to Stage 3 (ADR 0016)."""

import math

import numpy as np

from invariant.stages.adversarial.chain import Start, StateChain, state_path
from invariant.stages.adversarial.diagnostic import stress_diagnostic
from invariant.stages.adversarial.state import volatility_state


def _panel(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gen = np.random.default_rng(seed)
    n = 4000
    regime = state_path(StateChain(share=0.3, duration=40.0), Start.STATIONARY, n, gen)
    returns = gen.standard_normal((n, 4)) * np.where(regime == 1, 0.03, 0.01)[:, None]
    dates = np.arange(300, n - 1, dtype=np.int64)
    states, _ = volatility_state(returns.mean(axis=1), dates)
    ridge = np.where(states == 1, -0.003, 0.002) + 0.01 * gen.standard_normal(len(dates))
    ew = returns[dates + 1].mean(axis=1)
    return ridge, ew, returns, dates


def test_reports_the_high_state_sharpe_with_an_interval() -> None:
    ridge, ew, returns, dates = _panel(1)
    d = stress_diagnostic(
        ridge, ew, returns, dates, n_boot=199, confidence=0.95, gen=np.random.default_rng(2)
    )
    states, _ = volatility_state(returns.mean(axis=1), dates)
    hi = ridge[states == 1]
    assert d.ridge_sharpe_high == hi.mean() / hi.std(ddof=1) * math.sqrt(252)
    assert d.ridge_ci[0] < d.ridge_sharpe_high < d.ridge_ci[1] < 0
    assert d.ridge_sharpe_low > 0
    assert abs(d.high_share - 0.3) < 1 / len(dates) + 1e-12 and d.block_length >= 2


def test_is_deterministic_given_the_generator() -> None:
    ridge, ew, returns, dates = _panel(3)
    a = stress_diagnostic(
        ridge, ew, returns, dates, n_boot=99, confidence=0.9, gen=np.random.default_rng(4)
    )
    b = stress_diagnostic(
        ridge, ew, returns, dates, n_boot=99, confidence=0.9, gen=np.random.default_rng(4)
    )
    assert a == b
