from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from invariant.stages.decay.te import gaussian_te, ksg_windows
from invariant.stages.decay.windows import window_bounds


@given(st.floats(-0.999, 0.999))
def test_gaussian_te_closed_form(r: float) -> None:
    te = float(gaussian_te(np.array([r]))[0])
    assert te == pytest.approx(-0.5 * np.log(1.0 - r * r), abs=1e-12)
    assert te == pytest.approx(np.log(np.cosh(np.arctanh(r))), rel=1e-9, abs=1e-12)
    assert te >= 0.0


def test_ksg_is_deterministic_under_seeds_and_detects_a_strong_link() -> None:
    rng = np.random.default_rng(0)
    n = 300
    z = rng.standard_normal((n, 2))
    x = rng.standard_normal(n)
    y = np.r_[0.8 * x[:150], np.zeros(150)] + z[:, 0] + rng.standard_normal(n)
    bounds = window_bounds(n, 150)
    a = ksg_windows(x, y, z, bounds, [1, 2])
    b = ksg_windows(x, y, z, bounds, [1, 2])
    np.testing.assert_array_equal(a.value, b.value)
    np.testing.assert_array_equal(a.null_sd, b.null_sd)
    assert a.p[0] < 0.01  # strong link in the first window
    assert a.value[0] - a.null_mean[0] > a.value[1] - a.null_mean[1]
    assert (a.null_sd > 0).all()


def test_ksg_needs_one_seed_per_window() -> None:
    x = np.zeros(100)
    with pytest.raises(ValueError):
        ksg_windows(x, x, np.zeros((100, 0)), window_bounds(100, 50), [1])


def test_ksg_runs_with_an_empty_conditioning_set() -> None:
    """Regression: an empty Z takes tigramite's block-shuffle path, whose ``_get_acf`` called
    ``np.corrcoef(..., ddof=0)`` and raised TypeError on numpy 2 (docs/adr/0011)."""
    rng = np.random.default_rng(3)
    x = rng.standard_normal(200)
    y = 0.5 * x + rng.standard_normal(200)
    out = ksg_windows(x, y, np.empty((200, 0)), window_bounds(200, 100), [4, 5])
    assert np.isfinite(out.value).all() and (out.null_sd > 0).all()
    assert out.p[0] < 0.05


@given(st.integers(0, 2**32 - 1), st.integers(20, 300))
def test_fixed_acf_matches_reference(seed: int, n: int) -> None:
    from invariant.stages.decay.te import CMIknnNumpy2

    s = np.random.default_rng(seed).standard_normal(n).cumsum()
    acf = CMIknnNumpy2()._get_acf(s)
    max_lag = int(max(5, 0.1 * n))
    ref = [1.0] + [
        float(
            np.sum((a - a.mean()) * (b - b.mean()))
            / np.sqrt(np.sum((a - a.mean()) ** 2) * np.sum((b - b.mean()) ** 2))
        )
        for a, b in ((s[k:], s[: n - k]) for k in range(1, max_lag + 1))
    ]
    np.testing.assert_allclose(acf, ref, rtol=1e-10, atol=1e-12)


def test_fixed_class_equals_tigramite_when_z_is_not_empty() -> None:
    from tigramite.independence_tests.cmiknn import CMIknn

    from invariant.stages.decay.te import CMIknnNumpy2

    rng = np.random.default_rng(8)
    arr = rng.standard_normal((4, 150))
    arr[1] += 0.4 * arr[0]
    xyz = np.array([0, 1, 2, 2])
    results = []
    for cls in (CMIknn, CMIknnNumpy2):
        t = cls(seed=21, workers=1)
        v = t.get_dependence_measure(arr.copy(), xyz)
        p, null = t.get_shuffle_significance(arr.copy(), xyz, v, return_null_dist=True)
        results.append((v, p, null))
    assert results[0][0] == results[1][0] and results[0][1] == results[1][1]
    np.testing.assert_array_equal(results[0][2], results[1][2])
