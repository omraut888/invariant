from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from invariant.stages.causal.bootstrap import (
    circular_block_indices,
    default_block_length,
    partial_corr,
    partial_corr_ci,
)


def _ols_partial_corr(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    a = np.column_stack([np.ones(len(x)), z])
    rx = x - a @ np.linalg.lstsq(a, x, rcond=None)[0]
    ry = y - a @ np.linalg.lstsq(a, y, rcond=None)[0]
    return float(rx @ ry / np.sqrt((rx @ rx) * (ry @ ry)))


@given(st.integers(1, 200), st.integers(1, 50), st.integers(0, 2**32 - 1))
def test_indices_are_wrapped_contiguous_blocks(n: int, block: int, seed: int) -> None:
    block = min(block, n)
    idx = circular_block_indices(n, block, np.random.default_rng(seed))
    assert idx.shape == (n,) and idx.min() >= 0 and idx.max() < n
    for start in range(0, n, block):
        chunk = idx[start : start + block]
        assert ((np.diff(chunk) % n) == 1).all()
    again = circular_block_indices(n, block, np.random.default_rng(seed))
    np.testing.assert_array_equal(idx, again)


def test_block_length_validated() -> None:
    with pytest.raises(ValueError):
        circular_block_indices(10, 11, np.random.default_rng(0))
    assert default_block_length(6000) == 19
    assert default_block_length(1) == 1


@settings(max_examples=30, deadline=None)
@given(st.integers(0, 3), st.integers(0, 2**32 - 1))
def test_partial_corr_matches_residual_regression(k: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((300, k)) * 5 + 3
    x = 0.4 * z.sum(axis=1) + rng.standard_normal(300) + 100.0
    y = 0.3 * x - 0.2 * z.sum(axis=1) + rng.standard_normal(300) * 0.01
    assert partial_corr(x, y, z) == pytest.approx(_ols_partial_corr(x, y, z), abs=1e-10)


def test_ci_is_reproducible_and_brackets_the_estimate() -> None:
    rng = np.random.default_rng(1)
    z = rng.standard_normal((500, 2))
    x = rng.standard_normal(500)
    y = 0.2 * x + z[:, 0] + rng.standard_normal(500)
    kw = {"n_boot": 199, "block_length": None, "confidence": 0.9}
    a = partial_corr_ci(x, y, z, rng=np.random.default_rng(7), **kw)  # type: ignore[arg-type]
    b = partial_corr_ci(x, y, z, rng=np.random.default_rng(7), **kw)  # type: ignore[arg-type]
    assert a == b
    assert a.low < a.estimate < a.high
    assert a.block_length == default_block_length(500)
    assert a.estimate == pytest.approx(_ols_partial_corr(x, y, z), abs=1e-12)


def test_replicate_equals_regression_on_resampled_rows() -> None:
    # The weighted-moment shortcut must equal re-running the regression on the resampled rows.
    rng = np.random.default_rng(3)
    z = rng.standard_normal((200, 1))
    x = rng.standard_normal(200)
    y = 0.5 * x + rng.standard_normal(200)
    ci = partial_corr_ci(
        x, y, z, n_boot=1, block_length=5, confidence=0.5, rng=np.random.default_rng(11)
    )
    idx = circular_block_indices(200, 5, np.random.default_rng(11))
    want = _ols_partial_corr(x[idx], y[idx], z[idx])
    assert ci.low == pytest.approx(want, abs=1e-10)


@pytest.mark.slow
def test_coverage_on_ar1_regressor() -> None:
    # x is AR(1) with rho .9 (Stage 1's planted signals); y = beta x + iid noise, so the true
    # partial correlation given z (independent noise) is known. Reported, with a loose floor:
    # block dependence beyond b makes the interval somewhat narrow on persistent regressors.
    n, reps, rho, beta = 2000, 300, 0.9, 0.05
    true_r = beta / np.sqrt(beta**2 + 1.0)  # Var(x) = 1
    rng = np.random.default_rng(20261018)
    hits = 0
    for _ in range(reps):
        e = rng.standard_normal(n + 200)
        x = np.empty(n + 200)
        x[0] = e[0]
        for t in range(1, n + 200):
            x[t] = rho * x[t - 1] + np.sqrt(1 - rho**2) * e[t]
        x = x[200:]
        y = beta * x + rng.standard_normal(n)
        z = rng.standard_normal((n, 1))
        ci = partial_corr_ci(x, y, z, n_boot=299, block_length=None, confidence=0.9, rng=rng)
        hits += ci.low <= true_r <= ci.high
    coverage = hits / reps
    print(f"90% block-bootstrap coverage on AR(1) rho=.9 regressor: {coverage:.3f}")
    assert 0.82 <= coverage <= 0.97
