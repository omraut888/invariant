r"""Out-of-sample portfolio metrics: Sharpe, the Ledoit-Wolf Sharpe-difference test, the
Deflated Sharpe Ratio, turnover and exposure (docs/adr/0013).

Ledoit & Wolf (2008), J. Empirical Finance 15(5), sections 2-3 (transcribed)
-------------------------------------------------------------------------
For return series i and n, ``v = (mu_i, mu_n, g_i, g_n)``, with g the uncentred second moments.
``Delta = f(v) = a/sqrt(c - a^2) - b/sqrt(d - b^2)``. The delta method uses the gradient
``(c/(c-a^2)^1.5, -d/(d-b^2)^1.5, -a/(2(c-a^2)^1.5), b/(2(d-b^2)^1.5))`` and gives
``s(Delta) = sqrt(grad' Psi grad / T)`` (their eq. 5). The bootstrap is a circular block bootstrap
with block size b. Each replicate's standard error uses the block-sum covariance
``Psi* = (1/l) sum_j f_j f_j'`` with ``f_j = b^{-1/2} sum_t y*_{(j-1)b+t}`` (Gotze and Kunsch),
and the replicates are studentised. Their test is two-sided and symmetric, with p-value
``(#{|Delta* - Delta| / s(Delta*) >= |Delta| / s(Delta)} + 1) / (M + 1)`` (their eq. 9).

**Departures, recorded in ADR 0013:**
- **One-sided** (``alternative="greater"``) for the kill rule. The p-value is
  ``(#{(Delta* - Delta)/s* >= Delta/s} + 1)/(M + 1)``.
- **The original data's Psi** uses the same block-sum estimator on the non-overlapping blocks
  of the original series. LW use a prewhitened quadratic-spectral HAC.
- **Fixed block size** ``ceil(T^{1/3})`` in place of LW's calibration algorithm 3.1.

The kill test's size is gated empirically on the benchmark's null scenario, not assumed.

Bailey & Lopez de Prado (2014), J. Portfolio Management 40(5), eqs. (1)-(2) (transcribed)
-------------------------------------------------------------------------------------
- ``SR0 = sqrt(V[SR_n]) * ((1 - g) Phi^-1(1 - 1/N) + g Phi^-1(1 - 1/(N e)))``, where g is the
  Euler-Mascheroni constant, N the number of independent trials and V the variance of the
  trials' Sharpe ratios.
- ``DSR = Phi((SR - SR0) sqrt(T - 1) / sqrt(1 - skew SR + (kurt - 1)/4 SR^2))``, with
  non-annualised SR, T observations, and kurtosis *not* in excess.
"""

from __future__ import annotations

import math
from typing import Final, Literal

import numpy as np
import numpy.typing as npt
from scipy import stats

__all__ = [
    "TRADING_DAYS",
    "deflated_sharpe_from_moments",
    "deflated_sharpe_ratio",
    "exposure_shares",
    "sharpe",
    "sharpe_difference_test",
    "turnover",
]

_FloatArray = npt.NDArray[np.float64]
TRADING_DAYS: Final = 252
_EULER_GAMMA: Final = 0.5772156649015329


def sharpe(r: _FloatArray, *, annualize: bool = True) -> float:
    """Mean over standard deviation (ddof 1). ``nan`` for a constant series (e.g. all cash)."""
    sd = float(np.std(r, ddof=1))
    if not sd > 0.0:
        return math.nan
    s = float(np.mean(r)) / sd
    return s * math.sqrt(TRADING_DAYS) if annualize else s


def _grad(v: _FloatArray) -> _FloatArray:
    a, b, c, d = v
    vi, vn = c - a * a, d - b * b
    return np.array(
        [c / vi**1.5, -d / vn**1.5, -0.5 * a / vi**1.5, 0.5 * b / vn**1.5], dtype=np.float64
    )


def _delta(v: _FloatArray) -> float:
    a, b, c, d = v
    return float(a / math.sqrt(c - a * a) - b / math.sqrt(d - b * b))


def _block_psi(y: _FloatArray, block: int) -> _FloatArray:
    """``(1/l) sum_j f_j f_j'`` over the l non-overlapping blocks of the centred rows of y."""
    n_blocks = len(y) // block
    if n_blocks < 2:
        raise ValueError("series too short for the block size")
    f = y[: n_blocks * block].reshape(n_blocks, block, -1).sum(axis=1) / math.sqrt(block)
    return np.asarray(f.T @ f / n_blocks, dtype=np.float64)


def _stat(ri: _FloatArray, rn: _FloatArray, block: int) -> tuple[float, float]:
    v = np.array([ri.mean(), rn.mean(), (ri * ri).mean(), (rn * rn).mean()])
    y = np.column_stack([ri, rn, ri * ri, rn * rn]) - v
    g = _grad(v)
    se = math.sqrt(max(float(g @ _block_psi(y, block) @ g), 0.0) / len(ri))
    return _delta(v), se


def sharpe_difference_test(
    ri: _FloatArray,
    rn: _FloatArray,
    rng: np.random.Generator,
    *,
    n_boot: int = 4999,
    block: int | None = None,
    alternative: Literal["greater", "two-sided"] = "greater",
) -> tuple[float, float]:
    """``(Delta, p)`` for ``H0: Sharpe(i) = Sharpe(n)`` (module docstring). Non-annualised Delta.

    ``alternative="greater"`` tests ``Sharpe(i) > Sharpe(n)``. A series with zero variance (an
    arm that stayed in cash) cannot beat anything: it returns ``p = 1``.
    """
    if len(ri) != len(rn):
        raise ValueError("series must be paired")
    t = len(ri)
    b = block if block is not None else math.ceil(t ** (1.0 / 3.0))
    if np.std(ri) == 0.0 or np.std(rn) == 0.0:
        return (math.nan, 1.0)
    delta, se = _stat(ri, rn, b)
    n_blocks = math.ceil(t / b)
    exceed = 0
    for _ in range(n_boot):
        starts = rng.integers(0, t, size=n_blocks)
        idx = (starts[:, None] + np.arange(b)[None, :]).ravel()[:t] % t
        d_star, se_star = _stat(ri[idx], rn[idx], b)
        if alternative == "greater":
            exceed += (d_star - delta) / se_star >= delta / se
        else:
            exceed += abs(d_star - delta) / se_star >= abs(delta) / se
    return delta, (exceed + 1) / (n_boot + 1)


def deflated_sharpe_from_moments(
    sr: float, n_obs: int, skew: float, kurt: float, *, n_trials: int, trial_sharpe_variance: float
) -> float:
    """DSR from a non-annualised Sharpe ``sr`` over ``n_obs`` observations, its skewness and
    (non-excess) kurtosis, and the trials' count and Sharpe variance (module docstring)."""
    if n_trials < 2:
        raise ValueError("the DSR needs at least two trials")
    sr0 = math.sqrt(max(trial_sharpe_variance, 0.0)) * (
        (1.0 - _EULER_GAMMA) * float(stats.norm.ppf(1.0 - 1.0 / n_trials))
        + _EULER_GAMMA * float(stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e)))
    )
    denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr
    return float(stats.norm.cdf((sr - sr0) * math.sqrt(n_obs - 1) / math.sqrt(denom)))


def deflated_sharpe_ratio(r: _FloatArray, *, n_trials: int, trial_sharpe_variance: float) -> float:
    """Bailey & Lopez de Prado (2014) DSR of series ``r``, selected from ``n_trials`` trials whose
    non-annualised Sharpe ratios have variance ``trial_sharpe_variance``."""
    sr = sharpe(r, annualize=False)
    if math.isnan(sr):
        return math.nan
    return deflated_sharpe_from_moments(
        sr,
        len(r),
        float(stats.skew(r)),
        float(stats.kurtosis(r, fisher=False)),
        n_trials=n_trials,
        trial_sharpe_variance=trial_sharpe_variance,
    )


def turnover(weights: _FloatArray) -> _FloatArray:
    """``||w_t - w_{t-1}||_1`` per day (the first day trades in from cash). Position drift is
    ignored."""
    prev = np.vstack([np.zeros((1, weights.shape[1])), weights[:-1]])
    return np.asarray(np.abs(weights - prev).sum(axis=1), dtype=np.float64)


def exposure_shares(weights: _FloatArray, *, tol: float = 1e-6) -> dict[str, float]:
    """Share of days fully invested (gross = 1), all cash (gross = 0), and in between."""
    gross = np.abs(weights).sum(axis=1)
    full = float(np.mean(gross >= 1.0 - tol))
    cash = float(np.mean(gross <= tol))
    return {"invested": full, "cash": cash, "partial": 1.0 - full - cash}
