r"""Non-overlapping windows over a Stage 1 link's design, and the per-window statistics.

Method
------
The design is Stage 1's time-aligned sample for the best link: rows ``i`` of ``x_i =
X_{t-lag}``, ``y_i = R_t`` and ``z_i`` (the deciding test's conditioning set at those lags),
from ``causal.pcmci.lagged_design``. It is cut into ``K = n // w`` consecutive windows of
``w`` rows each. The windows are **aligned to the end** of the sample, so the most recent
window is always complete. The first ``n - K w`` rows go into no window. They are still used
by the decline test, which works on the whole design.

Within a window, x and y are each regressed on ``[1, z]`` *using that window's rows only*. The
window statistic is the correlation of the two residuals, i.e. the partial correlation of x
and y given z on that window. A window's value therefore depends only on its own rows
(tests/leakage/test_decay_window_inputs.py). Its Fisher transform ``atanh(r)`` has approximate
variance ``1 / (w - k - 3)`` for ``k`` conditioning columns (Fisher 1924), given i.i.d.
Gaussian rows.

Why non-overlapping windows: the spec's overlapping 60/120/252-day rolling windows give
estimates that are strongly correlated with each other. A curve fitted to them as if they were
independent would report CIs that are far too narrow. At the benchmark's default SNR, the
spec's 60- and 120-day windows also carry almost no information (docs/adr/0011).

Assumptions: within a window, the rows are weakly dependent and close to Gaussian. The
benchmark's returns have GARCH-t residuals, and its regressors are persistent (rho .9 to .99).
So the nominal ``1/(w - k - 3)`` variance is an approximation, and the study measures what it
costs in CI coverage. Every window shares the full-sample conditioning set chosen by Stage 1.

Reference: Fisher (1924), "The distribution of the partial correlation coefficient", Metron 3.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = ["WindowPartialCorr", "window_bounds", "window_partial_corr"]

_FloatArray = npt.NDArray[np.float64]
_IntArray = npt.NDArray[np.int64]


def window_bounds(n: int, width: int) -> _IntArray:
    """``(K, 2)`` array of ``[start, stop)`` row ranges: ``K = n // width`` end-aligned windows.

    Raises ``ValueError`` if ``width < 2`` or ``n < width``.
    """
    if width < 2:
        raise ValueError(f"width must be >= 2, got {width}")
    k = n // width
    if k < 1:
        raise ValueError(f"{n} rows hold no window of {width}")
    starts = n - k * width + width * np.arange(k, dtype=np.int64)
    return np.column_stack([starts, starts + width])


@dataclass(frozen=True)
class WindowPartialCorr:
    """Per-window partial correlations. ``fisher_var[j] = 1 / (w_j - k - 3)``."""

    bounds: _IntArray
    r: _FloatArray
    fisher_z: _FloatArray
    fisher_var: _FloatArray


def _residual(v: _FloatArray, design: _FloatArray) -> _FloatArray:
    coef, *_ = np.linalg.lstsq(design, v, rcond=None)
    return np.asarray(v - design @ coef, dtype=np.float64)


def window_partial_corr(
    x: _FloatArray, y: _FloatArray, z: _FloatArray, bounds: _IntArray
) -> WindowPartialCorr:
    """Partial correlation of ``x`` and ``y`` given ``[1, z]`` within each window of ``bounds``.

    ``z`` is ``(n, k)``, and ``k`` may be 0. A window with too few rows for the Fisher variance
    (``w <= k + 3``) raises ``ValueError``. A constant residual gives ``r = 0``.
    """
    n = len(x)
    if len(y) != n or z.shape[0] != n:
        raise ValueError("x, y and z must have the same number of rows")
    k = z.shape[1]
    rs = np.empty(len(bounds))
    var = np.empty(len(bounds))
    for j, (a, b) in enumerate(bounds):
        w = int(b - a)
        if w <= k + 3:
            raise ValueError(f"window of {w} rows is too short for {k} conditioning columns")
        design = np.column_stack([np.ones(w), z[a:b]])
        ex, ey = _residual(x[a:b], design), _residual(y[a:b], design)
        denom = float(np.sqrt(np.dot(ex, ex) * np.dot(ey, ey)))
        rs[j] = float(np.dot(ex, ey)) / denom if denom > 0.0 else 0.0
        var[j] = 1.0 / (w - k - 3)
    rs = np.clip(rs, -1.0 + 1e-12, 1.0 - 1e-12)
    return WindowPartialCorr(bounds=bounds, r=rs, fisher_z=np.arctanh(rs), fisher_var=var)
