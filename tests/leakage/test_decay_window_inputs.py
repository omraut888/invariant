"""A Stage 2 window statistic depends only on the rows inside that window.

So the edge measured for a window never uses later data. The half-life fit and the decline test
are full-sample statements about the sample, like Stage 1's verdict (docs/adr/0011).
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from invariant.stages.decay.te import ksg_windows
from invariant.stages.decay.windows import window_bounds, window_partial_corr


def _data(seed: int, n: int, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, k))
    x = rng.standard_normal(n)
    return x, 0.2 * x + rng.standard_normal(n), z


@settings(max_examples=25, deadline=None)
@given(st.integers(0, 2**32 - 1), st.integers(0, 3), st.integers(1, 5))
def test_partial_corr_window_ignores_rows_after_it(seed: int, k: int, j: int) -> None:
    x, y, z = _data(seed, 600, k)
    bounds = window_bounds(600, 100)
    before = window_partial_corr(x, y, z, bounds)
    cut = int(bounds[j - 1, 1])  # perturb everything after window j-1
    rng = np.random.default_rng(seed + 1)
    x2, y2, z2 = x.copy(), y.copy(), z.copy()
    x2[cut:] += rng.standard_normal(600 - cut) * 5
    y2[cut:] = rng.standard_normal(600 - cut)
    z2[cut:] *= -3.0
    after = window_partial_corr(x2, y2, z2, bounds)
    np.testing.assert_array_equal(before.r[:j], after.r[:j])


def test_ksg_window_ignores_rows_after_it() -> None:
    x, y, z = _data(7, 240, 2)
    bounds = window_bounds(240, 80)
    seeds = [11, 12, 13]
    before = ksg_windows(x, y, z, bounds, seeds)
    y2 = y.copy()
    y2[160:] = np.random.default_rng(3).standard_normal(80)
    after = ksg_windows(x, y2, z, bounds, seeds)
    np.testing.assert_array_equal(before.value[:2], after.value[:2])
    np.testing.assert_array_equal(before.null_mean[:2], after.null_mean[:2])
