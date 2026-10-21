from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from invariant.stages.decay.decline import break_grid, decline_test, sup_split_statistic


def _hc0_split_t(x: np.ndarray, y: np.ndarray, k: int) -> float:
    """Brute force: separate OLS (no intercept) and White HC0 variance in each segment."""
    out = []
    for seg in (slice(0, k), slice(k, None)):
        xs, ys = x[seg], y[seg]
        b = xs @ ys / (xs @ xs)
        e = ys - b * xs
        out.append((b, np.sum(xs**2 * e**2) / (xs @ xs) ** 2))
    (b1, v1), (b2, v2) = out
    return float((b1 - b2) / np.sqrt(v1 + v2))


@settings(max_examples=30, deadline=None)
@given(st.integers(0, 2**32 - 1))
def test_cumulative_sum_statistic_matches_brute_force(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = 400
    x = rng.standard_normal(n)
    y = 0.3 * x + rng.standard_normal(n) * (1 + np.arange(n) / n)
    ks = break_grid(n, 0.15, 7)
    expected = max(_hc0_split_t(x, y, int(k)) for k in ks)
    assert float(sup_split_statistic(x, y, ks)) == pytest.approx(expected, rel=1e-9)


@settings(max_examples=20, deadline=None)
@given(st.integers(0, 2**32 - 1), st.floats(-2.0, 2.0))
def test_statistic_invariant_to_a_common_coefficient(seed: int, beta: float) -> None:
    rng = np.random.default_rng(seed)
    n = 500
    x, y = rng.standard_normal(n), rng.standard_normal(n)
    ks = break_grid(n, 0.15, 10)
    a = float(sup_split_statistic(x, y, ks))
    b = float(sup_split_statistic(x, y + beta * x, ks))
    assert a == pytest.approx(b, rel=1e-8, abs=1e-8)


def test_decline_is_detected_and_deterministic() -> None:
    rng = np.random.default_rng(1)
    n = 3000
    z = rng.standard_normal((n, 2))
    x = rng.standard_normal(n) + 0.5 * z[:, 0]
    beta = np.where(np.arange(n) < 1800, 0.3, 0.0)
    y = beta * x + z[:, 1] + rng.standard_normal(n)
    a = decline_test(x, y, z, np.random.default_rng(9), n_boot=199)
    b = decline_test(x, y, z, np.random.default_rng(9), n_boot=199)
    assert a == b
    assert a.p == pytest.approx(1 / 200) and a.statistic > 5
    assert abs(a.break_fraction - 0.6) < 0.05
    assert a.b_pre > a.b_post and a.trend_t < 0 and a.trend_p < 0.01


def test_rising_edge_is_not_a_decline() -> None:
    rng = np.random.default_rng(2)
    n = 3000
    x = rng.standard_normal(n)
    y = np.where(np.arange(n) < 1500, 0.0, 0.3) * x + rng.standard_normal(n)
    res = decline_test(x, y, np.empty((n, 0)), np.random.default_rng(0), n_boot=199)
    assert res.p > 0.5


@pytest.mark.slow
def test_size_under_constant_coefficient_with_heteroskedasticity() -> None:
    """400 replications: the rejection rate at 5% should be within about +-2.6 points."""
    rng = np.random.default_rng(20261021)
    n, rejections = 2000, 0
    for _ in range(400):
        x = rng.standard_normal(n)
        sd = np.exp(0.5 * np.sin(np.arange(n) / 150.0))  # slow volatility cycle
        y = 0.1 * x + sd * rng.standard_t(6, n) / np.sqrt(1.5)
        rejections += decline_test(x, y, np.empty((n, 0)), rng, n_boot=199).p < 0.05
    assert 0.024 <= rejections / 400 <= 0.076


def test_break_grid_validated() -> None:
    with pytest.raises(ValueError):
        break_grid(100, 0.0, 1)
    with pytest.raises(ValueError):
        break_grid(100, 0.15, 0)
    ks = break_grid(1000, 0.15, 25)
    assert ks[0] == 150 and ks[-1] <= 850
