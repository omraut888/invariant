r"""Stage 2's kill test: has the edge of a Stage 1 link declined? (docs/adr/0011)

Method
------
On Stage 1's design ``(x, y, z)`` for the best link, x and y are first residualised on ``[1, z]``
with full-sample coefficients, giving ``x~`` and ``y~``. This is the *partial* structural-change
model (Bai & Perron 2003), in which only the signal's coefficient may break and the
conditioning coefficients stay constant. For every candidate break row ``k`` on a grid over
``[trim n, (1 - trim) n]``:

    b_pre(k)  = sum_{i<k}  x~ y~ / sum_{i<k}  x~^2,   b_post(k) likewise over i >= k
    t(k)      = (b_pre - b_post) / sqrt(V_pre + V_post)

``V`` is White's (1980) HC0 variance of each segment's coefficient. Every segment sum is a
difference of cumulative sums, so the whole grid costs O(n). The statistic is the one-sided
``S = max_k t(k)`` (Andrews 1993): a large S means the edge was stronger before some break
than after it. S makes no assumption about where a decline starts or what shape it has.

p-value: Hansen's (2000) fixed-regressor bootstrap. Each replicate keeps ``x~`` fixed and sets
``y*_i = e_i eta_i``, where ``e`` holds the residuals of the no-break fit and ``eta`` is i.i.d.
Rademacher. That keeps each row's heteroskedasticity. A common coefficient would shift both
b's equally, so the null of "no break" is the only null that needs simulating. Then
``p = (1 + #{S* >= S}) / (B + 1)``.

Kill rule: ``p < alpha``. The break row at the maximum is reported. It is descriptive and is
not an estimate of the decay's onset.

A diagnostic that never decides: the one-sided HC0 t of the trend interaction ``x~ * (i/n -
1/2)`` in ``y~ ~ x~ + x~ (i/n - 1/2)``, with a normal reference.

Assumptions and limitations
---------------------------
- The residuals are a martingale difference sequence given the regressors. That holds when
  Stage 1's conditioning set contains every persistent parent of R. A persistent omitted
  parent (e.g. a hidden confounder) makes the errors serially correlated, and then the wild
  bootstrap understates the variance. The study measures this on ``hidden_confounder``.
- HC0, not HC3: the leverage of one row within a segment of at least ``trim n`` rows (900 at
  n = 6000) is about ``1/(trim n)``, so HC3's ``(1 - h)^-2`` correction changes V by well under
  1%. The bootstrap calibrates the statistic either way.
- Conditioning coefficients are estimated on the full sample and held fixed. A break in them
  that is correlated with the signal's would leak into ``t(k)``.
- Power is lowest for a decline that started recently: a short post-break segment carries
  little information (docs/adr/0011, finding 4).

References: Andrews (1993), "Tests for parameter instability and structural change with
unknown change point", Econometrica 61(4); Hansen (2000), "Testing for structural change in
conditional models", J. Econometrics 97(1); Bai & Perron (2003), J. Applied Econometrics
18(1); White (1980), Econometrica 48(4).
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field
from scipy import stats

__all__ = ["DeclineTest", "break_grid", "decline_test", "sup_split_statistic"]

_FloatArray = npt.NDArray[np.float64]
_IntArray = npt.NDArray[np.int64]
_CHUNK: Final = 64


class DeclineTest(BaseModel):
    """The decline test of one link. Coefficients are on the residualised scale."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    statistic: float
    p: float = Field(ge=0.0, le=1.0)
    n_boot: int = Field(ge=1)
    n_rows: int = Field(ge=1)
    break_row: int
    break_fraction: float = Field(ge=0.0, le=1.0)
    b_full: float
    b_pre: float
    b_post: float
    se_pre: float = Field(ge=0.0)
    se_post: float = Field(ge=0.0)
    trend_t: float
    trend_p: float = Field(ge=0.0, le=1.0)
    """One-sided trend-interaction diagnostic (never decides)."""


def break_grid(n: int, trim: float, step: int) -> _IntArray:
    """Break rows ``k`` (first post-break row) on ``[ceil(trim n), floor((1 - trim) n)]``."""
    if not 0.0 < trim < 0.5:
        raise ValueError(f"trim must be in (0, 0.5), got {trim}")
    lo, hi = math.ceil(trim * n), math.floor((1.0 - trim) * n)
    if hi <= lo or step < 1:
        raise ValueError(f"no break rows for n={n}, trim={trim}, step={step}")
    return np.arange(lo, hi + 1, step, dtype=np.int64)


def _residualise(v: _FloatArray, design: _FloatArray) -> _FloatArray:
    coef, *_ = np.linalg.lstsq(design, v, rcond=None)
    return np.asarray(v - design @ coef, dtype=np.float64)


def _segment_t(
    x: _FloatArray, y: _FloatArray, ks: _IntArray
) -> tuple[_FloatArray, _FloatArray, _FloatArray, _FloatArray, _FloatArray]:
    """``t(k)``, b_pre, b_post, V_pre, V_post for ``y`` of shape ``(..., n)``."""
    x2 = x * x
    c_xx = np.cumsum(x2)
    c_x4 = np.cumsum(x2 * x2)
    c_xy = np.cumsum(x * y, axis=-1)
    c_x3y = np.cumsum(x2 * x * y, axis=-1)
    c_x2y2 = np.cumsum(x2 * y * y, axis=-1)
    i = ks - 1

    def seg(c: _FloatArray, pre: bool) -> _FloatArray:
        total = c[..., -1:]
        return c[..., i] if pre else total - c[..., i]

    out = []
    for pre in (True, False):
        sxx, sx4 = seg(c_xx, pre), seg(c_x4, pre)
        sxy, sx3y, sx2y2 = seg(c_xy, pre), seg(c_x3y, pre), seg(c_x2y2, pre)
        b = sxy / sxx
        meat = np.maximum(sx2y2 - 2.0 * b * sx3y + b * b * sx4, 0.0)
        out.append((b, meat / (sxx * sxx)))
    (b_pre, v_pre), (b_post, v_post) = out
    t = (b_pre - b_post) / np.sqrt(np.maximum(v_pre + v_post, 1e-300))
    return t, b_pre, b_post, v_pre, v_post


def sup_split_statistic(x: _FloatArray, y: _FloatArray, ks: _IntArray) -> _FloatArray:
    """``max_k t(k)`` for residualised ``x`` (n,) and ``y`` (..., n)."""
    t, *_ = _segment_t(x, y, ks)
    return np.asarray(np.max(t, axis=-1), dtype=np.float64)


def decline_test(
    x: _FloatArray,
    y: _FloatArray,
    z: _FloatArray,
    rng: np.random.Generator,
    *,
    n_boot: int = 999,
    trim: float = 0.15,
    step: int = 25,
) -> DeclineTest:
    """One-sided sup-split decline test with a fixed-regressor wild bootstrap (module doc)."""
    n = len(x)
    if len(y) != n or z.shape[0] != n:
        raise ValueError("x, y and z must have the same number of rows")
    design = np.column_stack([np.ones(n), z])
    xr, yr = _residualise(x, design), _residualise(y, design)
    ks = break_grid(n, trim, step)
    t, b_pre, b_post, v_pre, v_post = _segment_t(xr, yr, ks)
    j = int(np.argmax(t))
    stat = float(t[j])
    b_full = float(np.dot(xr, yr) / np.dot(xr, xr))
    resid = yr - b_full * xr
    exceed = 0
    for start in range(0, n_boot, _CHUNK):
        size = min(_CHUNK, n_boot - start)
        eta = rng.integers(0, 2, size=(size, n)).astype(np.float64) * 2.0 - 1.0
        exceed += int(np.sum(sup_split_statistic(xr, resid * eta, ks) >= stat))
    trend_t = _trend_t(xr, yr)
    return DeclineTest(
        statistic=stat,
        p=(1 + exceed) / (n_boot + 1),
        n_boot=n_boot,
        n_rows=n,
        break_row=int(ks[j]),
        break_fraction=float(ks[j]) / n,
        b_full=b_full,
        b_pre=float(b_pre[j]),
        b_post=float(b_post[j]),
        se_pre=math.sqrt(float(v_pre[j])),
        se_post=math.sqrt(float(v_post[j])),
        trend_t=trend_t,
        trend_p=float(stats.norm.cdf(trend_t)),
    )


def _trend_t(x: _FloatArray, y: _FloatArray) -> float:
    """HC0 t of the trend interaction in ``y ~ x + x (i/n - 1/2)``. Negative = declining."""
    n = len(x)
    design = np.column_stack([x, x * (np.arange(n) / n - 0.5)])
    xtx_inv = np.linalg.inv(design.T @ design)
    coef = xtx_inv @ design.T @ y
    e = y - design @ coef
    meat = (design * (e * e)[:, None]).T @ design
    var = xtx_inv @ meat @ xtx_inv
    return float(coef[1] / math.sqrt(var[1, 1]))
