"""Every CI test used for decisions must be invariant to affine transforms of X, Y and Z.

A partial-correlation test that changes its p-value when a constant is added to a variable is
broken (ADR 0009, the ParCorrWLS defect). The defective classes are pinned here too, so a
tigramite fix shows up as a failing "still defective" test rather than silently.
"""

import numpy as np
import pytest
import statsmodels.api as sm
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats
from tigramite import data_processing as pp

from invariant.stages.causal.ci_tests import CITestName, hc3_p_value, make_ci_test

DECISION_TESTS = [
    CITestName.PARCORR,
    CITestName.ROBUST_PARCORR,
    CITestName.PARCORR_HC3,
    CITestName.PARCORR_WLS_CENTERED,
    CITestName.PARCORR_WLS_CENTERED_LOO,
]
DEFECTIVE = [CITestName.PARCORR_WLS, CITestName.PARCORR_WLS_LOO]
Z_SETS = {"empty": [], "other": [(2, -1)], "own_lag": [(0, -2)]}


def _data(seed: int, t: int = 400) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(t)
    z = rng.standard_normal(t)
    h = 1.0 + np.abs(np.r_[0.0, x[:-1]])  # heteroskedastic in X, as in E4
    y = 0.1 * np.r_[0.0, x[:-1]] + 0.3 * z + rng.standard_normal(t) * h
    return np.column_stack([x, y, z])


def _p(name: CITestName, data: np.ndarray, z: list[tuple[int, int]]) -> float:
    test = make_ci_test(name, seed=0, wls_window=10)
    test.set_dataframe(pp.DataFrame(data))
    _, p = test.run_test(X=[(0, -1)], Y=[(1, 0)], Z=z, tau_max=2)
    return float(p)


_scale = st.one_of(st.floats(0.01, 100.0), st.floats(-100.0, -0.01))
_shift = st.floats(-1e3, 1e3)


@pytest.mark.parametrize("name", DECISION_TESTS)
@pytest.mark.parametrize("z_set", list(Z_SETS))
@settings(max_examples=15, deadline=None)
@given(
    seed=st.integers(0, 2**16),
    scales=st.tuples(_scale, _scale, _scale),
    shifts=st.tuples(_shift, _shift, _shift),
)
def test_p_value_is_affine_invariant(
    name: CITestName,
    z_set: str,
    seed: int,
    scales: tuple[float, float, float],
    shifts: tuple[float, float, float],
) -> None:
    data = _data(seed)
    moved = data * np.array(scales) + np.array(shifts)
    if name is CITestName.ROBUST_PARCORR and any(s < 0 for s in scales):
        # Rank-based: a sign flip reverses ranks, so normal scores are mirrored but not
        # bit-identical (ties in the normal-score grid). Positive scales only.
        moved = data * np.abs(np.array(scales)) + np.array(shifts)
    p0, p1 = _p(name, data, Z_SETS[z_set]), _p(name, moved, Z_SETS[z_set])
    assert p1 == pytest.approx(p0, rel=1e-6, abs=1e-10)


@pytest.mark.parametrize("name", DEFECTIVE)
def test_defective_wls_is_still_location_dependent(name: CITestName) -> None:
    """tigramite 5.2.10.1's ParCorrWLS defect. If this fails, re-check the upstream fix."""
    data = _data(3)
    moved = data + np.array([10.0, 0.0, 0.0])
    # Invariant tests agree to 1e-6 (above); a 1e-3 gap proves location dependence.
    assert abs(_p(name, data, []) - _p(name, moved, [])) > 1e-3


@pytest.mark.parametrize("n_z", [0, 1, 3])
def test_hc3_matches_statsmodels(n_z: int) -> None:
    rng = np.random.default_rng(n_z)
    t = 300
    x = rng.standard_normal(t)
    zs = rng.standard_normal((t, n_z))
    y = 0.15 * x + zs.sum(axis=1) * 0.2 + rng.standard_normal(t) * (1 + x**2)
    fit = sm.OLS(y, sm.add_constant(np.column_stack([x, zs]))).fit(cov_type="HC3")
    expected = 2 * stats.t.sf(abs(fit.tvalues[1]), t - 2 - n_z)
    assert hc3_p_value(np.vstack([x, y, zs.T])) == pytest.approx(expected, rel=1e-10)


def test_hc3_value_is_parcorr_partial_correlation() -> None:
    data = _data(5)
    vals = []
    for name in (CITestName.PARCORR, CITestName.PARCORR_HC3):
        test = make_ci_test(name, seed=0)
        test.set_dataframe(pp.DataFrame(data))
        vals.append(test.run_test(X=[(0, -1)], Y=[(1, 0)], Z=[(2, -1)], tau_max=2)[0])
    assert vals[0] == vals[1]
