"""Observable volatility state for Stage 4 (docs/adr/0016, "Decision: an observable volatility
state").

Method
------
``v_t`` is the sample sd (ddof 1) of the equal-weight daily return over days ``t-20..t``. Day t is
in the high-volatility state iff ``v_t`` exceeds the q-quantile (q = .70) of ``v`` over the
decision dates of Stage 3's out-of-sample path. The P&L earned on day t+1 (weights dated t) is
labelled with the state of day t, so a label is fixed before the return it labels.

Why observable and not latent: a Markov-switching GARCH-t labeller could not identify the
benchmark's planted regimes from the equal-weight return (ADR 0016, Result). The state is a
volatility *state*, not a recovered regime. On the benchmark it overlaps the true regime only
partly, and that overlap is reported, not assumed.

Assumptions: the quantile is in-sample over the evaluated days, so the state share is exactly
1 - q there by construction. The trailing window needs 21 prior days, which every Stage 3
decision date has (window 1260).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

__all__ = ["trailing_volatility", "volatility_state"]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def trailing_volatility(ew: FloatArray, window: int = 21) -> FloatArray:
    """``v_t`` = sd (ddof 1) of ``ew[t-window+1..t]``; NaN for the first ``window - 1`` days."""
    x = np.asarray(ew, dtype=np.float64)
    if window < 2:
        raise ValueError("window must be at least 2")
    out = np.full(len(x), np.nan)
    if len(x) < window:
        return out
    views = np.lib.stride_tricks.sliding_window_view(x, window)
    out[window - 1 :] = views.std(axis=1, ddof=1)
    return out


def volatility_state(
    ew: FloatArray,
    dates: npt.NDArray[np.int64],
    *,
    quantile: float = 0.70,
    window: int = 21,
) -> tuple[IntArray, float]:
    """``(state, threshold)``: 1 where ``v[dates]`` exceeds its ``quantile``, else 0.

    ``dates`` are Stage 3's decision dates (``ArmPath.dates``); ``state[k]`` labels the P&L row
    ``k``, earned on ``dates[k] + 1``. Raises if any decision date lacks a full trailing window.
    """
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be in (0, 1)")
    v = trailing_volatility(ew, window)[dates]
    if not np.all(np.isfinite(v)):
        raise ValueError(f"every decision date needs {window} trailing returns")
    threshold = float(np.quantile(v, quantile))
    return (v > threshold).astype(np.int64), threshold
