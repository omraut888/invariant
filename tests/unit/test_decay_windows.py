from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from invariant.stages.decay.windows import window_bounds, window_partial_corr


def _ols_partial_corr(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    a = np.column_stack([np.ones(len(x)), z])
    rx = x - a @ np.linalg.lstsq(a, x, rcond=None)[0]
    ry = y - a @ np.linalg.lstsq(a, y, rcond=None)[0]
    return float(rx @ ry / np.sqrt((rx @ rx) * (ry @ ry)))


@given(st.integers(2, 5000), st.integers(2, 400))
def test_bounds_are_end_aligned_contiguous_windows(n: int, width: int) -> None:
    if n < width:
        with pytest.raises(ValueError):
            window_bounds(n, width)
        return
    b = window_bounds(n, width)
    assert len(b) == n // width
    assert b[-1, 1] == n
    assert (b[:, 1] - b[:, 0] == width).all()
    assert (b[1:, 0] == b[:-1, 1]).all()
    assert b[0, 0] == n - (n // width) * width


def test_bounds_validated() -> None:
    with pytest.raises(ValueError):
        window_bounds(100, 1)


@settings(max_examples=30, deadline=None)
@given(st.integers(0, 3), st.integers(0, 2**32 - 1))
def test_window_r_equals_ols_partial_correlation(k: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    n, w = 300, 60
    z = rng.standard_normal((n, k))
    x = rng.standard_normal(n) + (z.sum(axis=1) if k else 0.0)
    y = 0.3 * x + rng.standard_normal(n)
    bounds = window_bounds(n, w)
    out = window_partial_corr(x, y, z, bounds)
    for j, (a, b) in enumerate(bounds):
        assert out.r[j] == pytest.approx(_ols_partial_corr(x[a:b], y[a:b], z[a:b]), abs=1e-10)
        assert out.fisher_var[j] == pytest.approx(1.0 / (w - k - 3))
    np.testing.assert_allclose(out.fisher_z, np.arctanh(out.r))


def test_window_too_short_for_conditioning_set() -> None:
    rng = np.random.default_rng(0)
    x, y, z = rng.standard_normal(20), rng.standard_normal(20), rng.standard_normal((20, 3))
    with pytest.raises(ValueError):
        window_partial_corr(x, y, z, window_bounds(20, 6))
