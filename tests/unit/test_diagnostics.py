import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats

from invariant.benchmark.diagnostics import CovType, newey_west_lags, ols


def test_iid_se_matches_scipy_linregress() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal(500)
    y = 0.5 * x + rng.standard_normal(500)
    res = ols(y, x)
    ref = stats.linregress(x, y)
    assert res.coef[1] == pytest.approx(ref.slope)
    assert res.se[1] == pytest.approx(ref.stderr)
    assert res.coef[0] == pytest.approx(ref.intercept)


def test_hac_with_zero_lags_is_white_hc0() -> None:
    rng = np.random.default_rng(1)
    x = rng.standard_normal((300, 2))
    y = x @ [1.0, -1.0] + rng.standard_normal(300) * (1 + np.abs(x[:, 0]))
    res = ols(y, x, cov=CovType.HAC, hac_lags=0)
    design = np.column_stack([np.ones(300), x])
    e = y - design @ res.coef
    bread = np.linalg.inv(design.T @ design)
    meat = (design * e[:, None] ** 2).T @ design
    np.testing.assert_allclose(res.se, np.sqrt(np.diag(bread @ meat @ bread)))
    assert res.hac_lags == 0


def test_hac_se_larger_than_iid_with_autocorrelated_regressor_and_errors() -> None:
    rng = np.random.default_rng(2)
    n = 4000

    def ar1(phi: float) -> np.ndarray:
        e = rng.standard_normal(n)
        out = np.empty(n)
        out[0] = e[0]
        for t in range(1, n):
            out[t] = phi * out[t - 1] + e[t]
        return out

    x, u = ar1(0.9), ar1(0.9)
    iid = ols(u, x)
    hac = ols(u, x, cov=CovType.HAC, hac_lags=40)
    assert hac.se[1] > 2 * iid.se[1]


def test_default_lag_rule() -> None:
    assert newey_west_lags(100) == 4
    assert newey_west_lags(6000) == 9
    with pytest.raises(ValueError):
        newey_west_lags(0)


def test_input_validation() -> None:
    with pytest.raises(ValueError, match="shape"):
        ols(np.zeros(5), np.zeros((4, 1)))
    with pytest.raises(ValueError, match="more observations"):
        ols(np.zeros(2), np.zeros((2, 1)))
    with pytest.raises(ValueError, match="rank"):
        ols(np.arange(10.0), np.ones(10))  # collinear with the constant
    with pytest.raises(ValueError, match="non-finite"):
        ols(np.array([1.0, np.nan, 2.0, 3.0]), np.arange(4.0))
    with pytest.raises(ValueError, match="hac_lags"):
        ols(np.arange(10.0), np.arange(10.0) ** 2, cov=CovType.HAC, hac_lags=-1)


@settings(max_examples=30)
@given(
    st.floats(-5, 5),
    st.floats(-5, 5),
    st.integers(20, 200),
)
def test_noiseless_data_recovers_coefficients(intercept: float, slope: float, n: int) -> None:
    x = np.linspace(-1, 1, n)
    res = ols(intercept + slope * x, x)
    np.testing.assert_allclose(res.coef, [intercept, slope], atol=1e-9)
