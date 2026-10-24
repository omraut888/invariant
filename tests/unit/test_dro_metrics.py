from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from invariant.stages.dro.metrics import (
    _block_psi,
    _delta,
    _grad,
    deflated_sharpe_from_moments,
    deflated_sharpe_ratio,
    exposure_shares,
    sharpe,
    sharpe_difference_test,
    turnover,
)


def test_dsr_reproduces_the_papers_worked_example() -> None:
    """Bailey & Lopez de Prado (2014): annualised SR 2.5 on 1250 daily obs (250/yr), V[SR] 1/2
    annualised, skew -3, kurtosis 10. DSR is ~.90 at N = 100 and .9505 at N = 46; with normal
    returns, .95 is reached at N = 88."""
    sr, v = 2.5 / math.sqrt(250), 0.5 / 250
    assert deflated_sharpe_from_moments(
        sr, 1250, -3, 10, n_trials=100, trial_sharpe_variance=v
    ) == pytest.approx(0.9004, abs=1e-4)
    assert deflated_sharpe_from_moments(
        sr, 1250, -3, 10, n_trials=46, trial_sharpe_variance=v
    ) == pytest.approx(0.9505, abs=1e-4)
    assert deflated_sharpe_from_moments(
        sr, 1250, 0, 3, n_trials=88, trial_sharpe_variance=v
    ) == pytest.approx(0.95, abs=1e-3)
    with pytest.raises(ValueError):
        deflated_sharpe_from_moments(sr, 1250, 0, 3, n_trials=1, trial_sharpe_variance=v)


@settings(max_examples=50, deadline=None)
@given(st.floats(-0.01, 0.01), st.floats(-0.01, 0.01), st.floats(1e-4, 1e-3), st.floats(1e-4, 1e-3))
def test_gradient_matches_finite_differences(a: float, b: float, vi: float, vn: float) -> None:
    v = np.array([a, b, vi + a * a, vn + b * b])
    g = _grad(v)
    h = 1e-9
    num = [(_delta(v + h * e) - _delta(v - h * e)) / (2 * h) for e in np.eye(4)]
    np.testing.assert_allclose(g, num, rtol=1e-4, atol=1e-6 * float(np.abs(g).max()))


def test_block_psi_with_block_one_is_the_sample_covariance() -> None:
    y = np.random.default_rng(0).standard_normal((500, 4))
    y -= y.mean(axis=0)
    np.testing.assert_allclose(_block_psi(y, 1), y.T @ y / 500)


def test_sharpe_difference_test_size_and_power() -> None:
    rng = np.random.default_rng(1)
    t = 1000
    null_p = []
    for _ in range(60):
        common = rng.standard_normal(t)
        ri = 0.03 + 0.7 * common + 0.7 * rng.standard_normal(t)
        rn = 0.03 + 0.7 * common + 0.7 * rng.standard_normal(t)
        null_p.append(sharpe_difference_test(ri, rn, rng, n_boot=199)[1])
    assert np.mean(np.array(null_p) < 0.05) <= 0.15  # 60 draws: loose size check
    ri = 0.15 + rng.standard_normal(t)
    rn = 0.0 + rng.standard_normal(t)
    delta, p = sharpe_difference_test(ri, rn, rng, n_boot=199)
    assert delta > 0 and p < 0.01
    _, p_rev = sharpe_difference_test(rn, ri, rng, n_boot=199)
    assert p_rev > 0.5
    _, p_two = sharpe_difference_test(ri, rn, rng, n_boot=199, alternative="two-sided")
    assert p_two < 0.02


def test_all_cash_never_beats_anything() -> None:
    rng = np.random.default_rng(2)
    assert sharpe_difference_test(np.zeros(300), rng.standard_normal(300), rng)[1] == 1.0
    assert math.isnan(sharpe(np.zeros(10)))
    assert math.isnan(deflated_sharpe_ratio(np.zeros(10), n_trials=5, trial_sharpe_variance=1e-4))


def test_turnover_and_exposure() -> None:
    w = np.array([[0.5, 0.5], [0.5, 0.5], [0.0, 0.0], [-1.0, 0.0], [0.2, 0.1]])
    np.testing.assert_allclose(turnover(w), [1.0, 0.0, 1.0, 1.0, 1.3])
    assert exposure_shares(w) == pytest.approx({"invested": 0.6, "cash": 0.2, "partial": 0.2})
    assert sharpe(np.array([0.01, 0.02, 0.0]), annualize=False) == pytest.approx(1.0)
