"""Cross-check ``invariant.benchmark.diagnostics`` against statsmodels (dev-only dependency).

``diagnostics.ols`` applies no small-sample correction to the HAC covariance, so the comparison
passes ``use_correction=False`` explicitly (the statsmodels 0.15 default, but stated so a change
of default cannot silently alter the reference). Both use the Bartlett kernel
``1 - l / (L + 1)``. ``test_tolerance_detects_small_sample_correction`` shows that the tolerance
is tight enough to detect the ``n / (n - k)`` correction (a ~0.25% SE difference here).
"""

import ast
from pathlib import Path

import numpy as np
import pytest
import statsmodels.api as sm

from invariant.benchmark.diagnostics import CovType, ols

RTOL = 1e-8
SRC = Path(__file__).resolve().parents[2] / "src" / "invariant"


def _data(seed: int, n: int = 800, k: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Persistent regressors; AR(1) errors whose scale depends on a regressor."""
    rng = np.random.default_rng(seed)
    x = np.empty((n, k))
    u = np.empty(n)
    shocks = rng.standard_normal((n, k + 1))
    x[0], u[0] = shocks[0, :k], shocks[0, k]
    for t in range(1, n):
        x[t] = 0.8 * x[t - 1] + shocks[t, :k]
        u[t] = 0.6 * u[t - 1] + shocks[t, k]
    errors = u * (0.5 + np.abs(x[:, 0]))  # heteroskedastic and autocorrelated
    y = 0.3 + x @ np.linspace(0.5, -0.5, k) + errors
    return y, x


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("add_constant", [True, False])
def test_ols_coefficients_and_iid_se_match(seed: int, add_constant: bool) -> None:
    y, x = _data(seed)
    design = sm.add_constant(x) if add_constant else x
    ref = sm.OLS(y, design).fit()
    res = ols(y, x, add_constant=add_constant)
    np.testing.assert_allclose(res.coef, ref.params, rtol=RTOL)
    np.testing.assert_allclose(res.se, ref.bse, rtol=RTOL)
    np.testing.assert_allclose(res.tstat, ref.tvalues, rtol=RTOL)
    assert res.nobs == int(ref.nobs)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("lags", [0, 1, 5, 12, 40])
def test_newey_west_se_match(seed: int, lags: int) -> None:
    y, x = _data(seed)
    ref = sm.OLS(y, sm.add_constant(x)).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": lags, "kernel": "bartlett", "use_correction": False},
    )
    res = ols(y, x, cov=CovType.HAC, hac_lags=lags)
    np.testing.assert_allclose(res.coef, ref.params, rtol=RTOL)
    np.testing.assert_allclose(res.se, ref.bse, rtol=RTOL)
    np.testing.assert_allclose(res.tstat, ref.tvalues, rtol=RTOL)
    assert res.hac_lags == lags


def test_default_bandwidth_matches_statsmodels_at_same_lag() -> None:
    y, x = _data(3)
    res = ols(y, x, cov=CovType.HAC)  # rule-of-thumb lag
    assert res.hac_lags is not None
    ref = sm.OLS(y, sm.add_constant(x)).fit(
        cov_type="HAC", cov_kwds={"maxlags": res.hac_lags, "use_correction": False}
    )
    np.testing.assert_allclose(res.se, ref.bse, rtol=RTOL)


def test_hac_zero_lags_matches_hc0() -> None:
    y, x = _data(4)
    ref = sm.OLS(y, sm.add_constant(x)).fit(cov_type="HC0")
    res = ols(y, x, cov=CovType.HAC, hac_lags=0)
    np.testing.assert_allclose(res.se, ref.bse, rtol=RTOL)


def test_tolerance_detects_small_sample_correction() -> None:
    y, x = _data(0)
    res = ols(y, x, cov=CovType.HAC, hac_lags=5)
    corrected = sm.OLS(y, sm.add_constant(x)).fit(
        cov_type="HAC", cov_kwds={"maxlags": 5, "use_correction": True}
    )
    n, k = len(y), x.shape[1] + 1
    np.testing.assert_allclose(res.se * np.sqrt(n / (n - k)), corrected.bse, rtol=RTOL)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(res.se, corrected.bse, rtol=RTOL)


def test_hac_differs_from_iid_on_this_data() -> None:
    # Guards against the cross-check being vacuous (e.g. errors accidentally i.i.d.).
    y, x = _data(0)
    iid = ols(y, x)
    hac = ols(y, x, cov=CovType.HAC, hac_lags=12)
    assert np.all(hac.se[1:] > 1.3 * iid.se[1:])


def test_library_code_never_imports_statsmodels() -> None:
    offenders = []
    for path in SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            if any(n == "statsmodels" or n.startswith("statsmodels.") for n in names):
                offenders.append(str(path.relative_to(SRC)))
    assert offenders == [], f"statsmodels is dev-only; imported in {offenders}"
