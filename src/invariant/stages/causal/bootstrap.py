r"""Circular block bootstrap CI for a partial correlation on a fixed, time-aligned design.

Method
------
The design is the rows ``t`` of ``(x_t, y_t, z_t)``, where for a Stage 1 link ``x_t = X_{t-tau}``,
``y_t = R_t`` and ``z_t`` is the decision test's conditioning set at those lags. Each row
already carries its own lags, so resampling rows keeps every within-row lag relation. Rows are
resampled in circular blocks of length ``b`` (Politis & Romano 1992), so that serial dependence
across rows up to about ``b`` is kept as well (Kunsch 1989). Each replicate re-estimates the
partial correlation, including the regression on ``z``. The interval is the percentile interval
of the replicates.

A replicate is a row multiset, so it equals a frequency-weighted regression on the original
rows. The replicates are therefore computed from weighted moment matrices ``M' diag(c) M`` with
``M = [1, z, x, y]``, one batched matrix product per chunk of replicates. The partial
correlation of ``x`` and ``y`` given ``[1, z]`` is ``-P_xy / sqrt(P_xx P_yy)`` with
``P = (M' diag(c) M)^{-1}``. The columns are standardised first, which leaves every partial
correlation unchanged and keeps the moment matrices well conditioned.

Default block length: ``ceil(n ** (1/3))``, the MSE-optimal rate of the block length for a
block-bootstrap variance estimate (Hall, Horowitz & Jing 1995). It is a rate, not a tuned
constant.

Assumptions and limitations:
- The rows are a stationary, weakly dependent sequence. A persistent regressor (rho .9 to .99)
  has dependence well beyond ``b``, which tends to make the interval too narrow. The coverage
  test (tests/unit/test_bootstrap.py, slow) measures how much on an AR(1) regressor.
- The conditioning set, and in Stage 1 the (asset, lag) at which the interval is taken, are
  held fixed at their values on the original sample. Both were selected on that sample, so the
  interval is not selection-adjusted. It describes the effect size at the selected link. It is
  not a test, and Stage 1 never decides with it.

References: Kunsch (1989), Annals of Statistics 17(3); Politis & Romano (1992), "A circular
block-resampling procedure for stationary data"; Hall, Horowitz & Jing (1995), Biometrika 82(3).
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "BootstrapCI",
    "circular_block_indices",
    "default_block_length",
    "partial_corr",
    "partial_corr_ci",
]

_FloatArray = npt.NDArray[np.float64]
_IntArray = npt.NDArray[np.int64]
_CHUNK = 100


class BootstrapCI(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    estimate: float
    low: float
    high: float
    confidence: float = Field(gt=0.0, lt=1.0)
    n_boot: int = Field(ge=1)
    block_length: int = Field(ge=1)
    n_rows: int = Field(ge=1)


def default_block_length(n: int) -> int:
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    b = math.ceil(math.cbrt(n))  # corrected below to the exact integer ceil(cbrt(n))
    while b**3 < n:
        b += 1
    while b > 1 and (b - 1) ** 3 >= n:
        b -= 1
    return b


def circular_block_indices(n: int, block: int, rng: np.random.Generator) -> _IntArray:
    """Row indices of one circular block-bootstrap resample of ``n`` rows."""
    if not 1 <= block <= n:
        raise ValueError(f"block length must be in [1, {n}], got {block}")
    n_blocks = -(-n // block)
    starts = rng.integers(0, n, size=n_blocks)
    idx = (starts[:, None] + np.arange(block)[None, :]) % n
    return np.asarray(idx.ravel()[:n], dtype=np.int64)


def _design(x: _FloatArray, y: _FloatArray, z: _FloatArray) -> _FloatArray:
    n = x.shape[0]
    if y.shape != (n,) or x.shape != (n,) or z.ndim != 2 or z.shape[0] != n:
        raise ValueError(f"shapes do not align: x {x.shape}, y {y.shape}, z {z.shape}")
    cols = np.column_stack([z, x, y])
    if not np.isfinite(cols).all():
        raise ValueError("design must be finite")
    sd = cols.std(axis=0)
    if (sd == 0).any():
        raise ValueError("a design column is constant")
    std = (cols - cols.mean(axis=0)) / sd
    return np.column_stack([np.ones(n), std])


def _from_moments(g: _FloatArray) -> _FloatArray:
    """Partial correlation of the last two columns given the others, per moment matrix."""
    p = np.linalg.inv(g)
    return np.asarray(-p[..., -2, -1] / np.sqrt(p[..., -2, -2] * p[..., -1, -1]), dtype=float)


def partial_corr(x: _FloatArray, y: _FloatArray, z: _FloatArray) -> float:
    """Partial correlation of ``x`` and ``y`` given ``[1, z]`` (``z`` is ``n x k``, k >= 0)."""
    m = _design(x, y, z)
    return float(_from_moments(m.T @ m))


def partial_corr_ci(
    x: _FloatArray,
    y: _FloatArray,
    z: _FloatArray,
    *,
    n_boot: int,
    block_length: int | None,
    confidence: float,
    rng: np.random.Generator,
) -> BootstrapCI:
    """Percentile circular-block-bootstrap interval of ``partial_corr(x, y, z)``."""
    m = _design(x, y, z)
    n = m.shape[0]
    block = default_block_length(n) if block_length is None else block_length
    estimate = float(_from_moments(m.T @ m))
    reps = np.empty(n_boot)
    for start in range(0, n_boot, _CHUNK):
        k = min(_CHUNK, n_boot - start)
        idx = np.stack([circular_block_indices(n, block, rng) for _ in range(k)])
        offsets = (np.arange(k) * n)[:, None]
        counts = np.bincount((idx + offsets).ravel(), minlength=k * n).reshape(k, n)
        g = np.einsum("bn,ni,nj->bij", counts.astype(float), m, m, optimize=True)
        reps[start : start + k] = _from_moments(g)
    if not np.isfinite(reps).all():
        raise RuntimeError("non-finite bootstrap replicate (singular resampled design)")
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(reps, [tail, 1.0 - tail])
    return BootstrapCI(
        estimate=estimate,
        low=float(low),
        high=float(high),
        confidence=confidence,
        n_boot=n_boot,
        block_length=block,
        n_rows=n,
    )
