"""FastParCorrWLS must be the published ParCorrWLS estimator, only faster."""

import numpy as np
import pytest
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.independence_tests.parcorr_wls import ParCorrWLS
from tigramite.independence_tests.robust_parcorr import RobustParCorr

from invariant.benchmark.config import load_benchmark_config
from invariant.benchmark.synthetic import generate_market
from invariant.stages.causal.ci_tests import CITestName, FastParCorrWLS, make_ci_test
from invariant.stages.causal.pcmci import run_pcmci


@pytest.mark.parametrize("dim", [2, 3, 6])
@pytest.mark.parametrize("standardize", [False, True])
@pytest.mark.parametrize("target_var", [0, 1])
def test_fast_residuals_match_tigramite(dim: int, standardize: bool, target_var: int) -> None:
    rng = np.random.default_rng(dim * 10 + standardize)
    t = 400
    array = rng.standard_normal((dim, t)) * rng.uniform(0.5, 2.0, (dim, 1))
    stds = rng.uniform(0.2, 3.0, (2, t))
    ref, fast = ParCorrWLS(), FastParCorrWLS()
    ref.stds = stds
    fast.stds = stds
    r_ref, m_ref = ref._get_single_residuals(array.copy(), target_var, standardize, True)
    r_fast, m_fast = fast._get_single_residuals(array.copy(), target_var, standardize, True)
    np.testing.assert_allclose(r_fast, r_ref, rtol=1e-12, atol=1e-12)
    if dim > 2:
        assert m_fast is not None
        np.testing.assert_allclose(m_fast, m_ref, rtol=1e-12, atol=1e-12)
    else:
        assert m_fast is None and m_ref is None


def test_fast_wls_pcmci_matches_tigramite() -> None:
    cfg = load_benchmark_config(None).model_copy(update={"n_obs": 800, "burn_in": 50})
    ds = generate_market(cfg, 3)
    frame = ds.features.join(ds.returns)
    data, names = frame.to_numpy(), list(frame.columns)
    ref = run_pcmci(data, names, ParCorrWLS(window_size=10), tau_max=3, pc_alpha=0.2)
    fast = run_pcmci(data, names, FastParCorrWLS(window_size=10), tau_max=3, pc_alpha=0.2)
    # Bit for bit, as the ci_tests docstring claims.
    np.testing.assert_array_equal(fast.p, ref.p)
    np.testing.assert_array_equal(fast.val, ref.val)


def test_make_ci_test_types() -> None:
    assert type(make_ci_test(CITestName.PARCORR, seed=1)) is ParCorr
    wls = make_ci_test(CITestName.PARCORR_WLS, seed=1, wls_window=25)
    assert isinstance(wls, FastParCorrWLS) and wls.window_size == 25
    assert type(make_ci_test(CITestName.ROBUST_PARCORR, seed=1)) is RobustParCorr
    with pytest.raises(ValueError, match="wls_window"):
        make_ci_test(CITestName.PARCORR_WLS, seed=1, wls_window=1)


def test_make_ci_test_returns_fresh_instances() -> None:
    a = make_ci_test(CITestName.PARCORR_WLS, seed=1)
    b = make_ci_test(CITestName.PARCORR_WLS, seed=1)
    assert a is not b


def test_leave_current_out_weights_exclude_current_residual() -> None:
    from invariant.stages.causal.ci_tests import LeaveCurrentOutParCorrWLS

    rng = np.random.default_rng(0)
    arr = rng.standard_normal((2, 200))  # dim 2: residual = |y|
    test = LeaveCurrentOutParCorrWLS(window_size=5)
    base = test._estimate_std_time(arr, 1)
    bumped = arr.copy()
    bumped[1, 100] += 10.0
    after = test._estimate_std_time(bumped, 1)
    changed = np.flatnonzero(after != base)
    assert changed.tolist() == [101, 102, 103, 104, 105]  # t+1..t+w, never t itself
    np.testing.assert_allclose(base[50], np.abs(arr[1, 45:50]).mean())
    np.testing.assert_allclose(base[:5], np.abs(arr[1, :5]).mean())
    # tigramite's estimator includes t
    assert (
        ParCorrWLS(window_size=5)._estimate_std_time(bumped, 1)[100]
        != (ParCorrWLS(window_size=5)._estimate_std_time(arr, 1)[100])
    )


def test_make_ci_test_loo() -> None:
    from invariant.stages.causal.ci_tests import LeaveCurrentOutParCorrWLS

    t = make_ci_test(CITestName.PARCORR_WLS_LOO, seed=1, wls_window=20)
    assert isinstance(t, LeaveCurrentOutParCorrWLS) and t.window_size == 20


@pytest.mark.parametrize("loo", [False, True])
def test_centered_wls_window_alignment(loo: bool) -> None:
    from invariant.stages.causal.ci_tests import CenteredParCorrWLS

    rng = np.random.default_rng(1)
    arr = rng.standard_normal((2, 200)) + np.array([[5.0], [-3.0]])  # nonzero means
    test = CenteredParCorrWLS(window_size=5, leave_current_out=loo)
    base = test._estimate_std_time(arr, 1)
    bumped = arr.copy()
    bumped[1, 100] += 10.0
    bumped[1, 160] -= 10.0  # mean-preserving, so the intercept (and every other residual) stays
    changed = np.flatnonzero(~np.isclose(test._estimate_std_time(bumped, 1), base))
    first = 101 if loo else 100  # leave-current-out: the window of t ends at t - 1
    expected = [*range(first, first + 5), *range(first + 60, first + 65)]
    assert changed.tolist() == expected
    resid = np.abs(arr[1] - arr[1].mean())
    window = resid[45:50] if loo else resid[46:51]
    np.testing.assert_allclose(base[50], window.mean())
    assert np.all(base > 0)


def test_make_ci_test_new_names() -> None:
    from invariant.stages.causal.ci_tests import CenteredParCorrWLS, ParCorrHC3

    assert isinstance(make_ci_test(CITestName.PARCORR_HC3, seed=1), ParCorrHC3)
    c = make_ci_test(CITestName.PARCORR_WLS_CENTERED_LOO, seed=1, wls_window=20)
    assert isinstance(c, CenteredParCorrWLS) and c.leave_current_out and c.window_size == 20
    with pytest.raises(ValueError, match="window_size"):
        make_ci_test(CITestName.PARCORR_WLS_CENTERED, seed=1, wls_window=1)
