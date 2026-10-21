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
