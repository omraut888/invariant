import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from invariant.benchmark.config import load_benchmark_config
from invariant.benchmark.stylized_facts import (
    StylizedFactsThresholds,
    acf,
    check_stylized_facts,
    robust_acf_se,
)
from invariant.benchmark.synthetic import generate_market


def _series(x: np.ndarray) -> "pd.Series[float]":
    return pd.Series(x, index=pd.bdate_range("2000-01-03", periods=len(x)))


def test_iid_gaussian_fails_tails_and_clustering_but_passes_acf() -> None:
    rng = np.random.default_rng(0)
    report = check_stylized_facts(_series(rng.standard_normal(6000) * 0.01))
    assert not report.check("heavy_tails").passed
    assert not report.check("volatility_clustering").passed
    assert report.check("no_linear_autocorrelation").passed
    assert not report.passed
    assert abs(report.excess_kurtosis) < 0.3


def test_synthetic_garch_t_returns_pass_all_facts() -> None:
    d = generate_market(load_benchmark_config(), 5)
    report = check_stylized_facts(d.returns["asset_00"])
    assert report.passed, report.checks


def test_ar1_returns_fail_linear_autocorrelation() -> None:
    rng = np.random.default_rng(1)
    e = rng.standard_normal(5000)
    x = np.empty_like(e)
    x[0] = e[0]
    for t in range(1, len(e)):
        x[t] = 0.3 * x[t - 1] + e[t]
    report = check_stylized_facts(_series(x))
    assert not report.check("no_linear_autocorrelation").passed
    assert report.acf[0] == pytest.approx(0.3, abs=0.05)


def test_report_carries_thresholds_and_lags() -> None:
    th = StylizedFactsThresholds(max_lag=5)
    rng = np.random.default_rng(2)
    report = check_stylized_facts(_series(rng.standard_normal(500)), th)
    assert report.thresholds == th
    assert len(report.acf) == len(report.abs_acf) == len(report.acf_robust_t) == 5
    assert report.n_obs == 500


def test_input_validation() -> None:
    rng = np.random.default_rng(3)
    with pytest.raises(ValueError, match="at least"):
        check_stylized_facts(_series(rng.standard_normal(100)))
    x = rng.standard_normal(1000)
    x[10] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        check_stylized_facts(_series(x))
    with pytest.raises(ValueError, match="constant"):
        check_stylized_facts(_series(np.ones(1000)))
    with pytest.raises(ValueError):
        check_stylized_facts(pd.Series(rng.standard_normal(1000)))  # no DatetimeIndex


def test_robust_se_matches_bartlett_for_iid_data() -> None:
    rng = np.random.default_rng(4)
    n = 20000
    se = robust_acf_se(rng.standard_normal(n), 5)
    np.testing.assert_allclose(se, 1 / np.sqrt(n), rtol=0.05)


def test_robust_se_exceeds_bartlett_under_garch() -> None:
    d = generate_market(load_benchmark_config(), 6)
    x = d.returns["asset_00"].to_numpy()
    assert robust_acf_se(x, 1)[0] > 1.2 / np.sqrt(len(x))


@settings(max_examples=25)
@given(st.lists(st.floats(-1, 1), min_size=30, max_size=200), st.floats(0.1, 100))
def test_acf_is_scale_and_shift_invariant(values: list[float], scale: float) -> None:
    x = np.array(values)
    if np.var(x) < 1e-6:
        return
    np.testing.assert_allclose(acf(x * scale + 3.0, 3), acf(x, 3), atol=1e-9)
    assert np.all(np.abs(acf(x, 3)) <= 1.0 + 1e-12)
