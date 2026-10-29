"""Automatic block-length selection for Stage 4's block bootstrap (docs/adr/0016).

Method
------
Politis & White (2004), "Automatic block-length selection for the dependent bootstrap",
Econometric Reviews 23(1), with the correction of Patton, Politis & White (2009), Econometric
Reviews 28(4). For a demeaned series of length n with sample autocovariances R(k):

1. **Bandwidth.** m_hat is the smallest lag after which K_N consecutive autocorrelations are all
   within c * sqrt(log10(n) / n), with K_N = max(5, ceil(sqrt(log10(n)))) and c = 2. Then
   M = min(2 m_hat, m_max), with m_max = ceil(sqrt(n)) + K_N.
2. **Flat-top kernel.** lambda(t) = 1 for |t| <= 1/2, 2(1 - |t|) for 1/2 < |t| <= 1, else 0.
3. **Estimates.** G = sum_{|k| <= M} lambda(k/M) |k| R(k) and g = sum_{|k| <= M} lambda(k/M) R(k).
   D_SB = 2 g^2 and D_CB = (4/3) g^2 (the 2009 correction).
4. **Block lengths.** b = (2 G^2 / D)^(1/3) n^(1/3), each capped at
   b_max = ceil(min(3 sqrt(n), n / 3)).

Stage 4 applies it to |r| of the equal-weight return and uses round(b_CB), floored at 2 (ADR 0016).

Assumptions: a stationary, weakly dependent series. The selector is an MSE-optimal rate-based
rule for the bootstrap variance of a mean. Using it to preserve volatility clustering in |r| is
an applied choice, recorded in the ADR.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

__all__ = ["BlockLengths", "optimal_block_length", "stage4_block_length"]

FloatArray = npt.NDArray[np.float64]


class BlockLengths(NamedTuple):
    stationary: float
    circular: float
    m_hat: int
    bandwidth: int


def _flat_top(t: FloatArray) -> FloatArray:
    a = np.abs(t)
    return np.where(a <= 0.5, 1.0, np.where(a <= 1.0, 2.0 * (1.0 - a), 0.0))


def optimal_block_length(x: FloatArray) -> BlockLengths:
    """Politis-White (2004) / Patton-Politis-White (2009) block lengths of ``x``."""
    y = np.asarray(x, dtype=np.float64)
    n = len(y)
    if y.ndim != 1 or n < 10 or not np.all(np.isfinite(y)):
        raise ValueError("x must be a finite 1-d series of at least 10 observations")
    y = y - y.mean()
    k_n = max(5, math.ceil(math.sqrt(math.log10(n))))
    m_max = math.ceil(math.sqrt(n)) + k_n
    b_max = math.ceil(min(3.0 * math.sqrt(n), n / 3.0))
    c = 2.0
    gamma0 = float(y @ y) / n
    if gamma0 == 0.0:
        raise ValueError("x is constant")
    acov = np.array([float(y[k:] @ y[: n - k]) / n for k in range(m_max + 1)])
    rho = acov[1:] / gamma0
    bound = c * math.sqrt(math.log10(n) / n)
    small = np.abs(rho) < bound
    m_hat = m_max
    for m in range(1, m_max - k_n + 2):
        if small[m - 1 : m - 1 + k_n].all():
            m_hat = m - 1
            break
    bw = min(2 * max(m_hat, 1), m_max)
    lags = np.arange(-bw, bw + 1)
    lam = _flat_top(lags / bw)
    r = acov[np.abs(lags)]
    g_big = float((lam * np.abs(lags) * r).sum())
    g0 = float((lam * r).sum())
    d_sb, d_cb = 2.0 * g0 * g0, (4.0 / 3.0) * g0 * g0
    root_n = n ** (1.0 / 3.0)
    b_sb = (2.0 * g_big * g_big / d_sb) ** (1.0 / 3.0) * root_n
    b_cb = (2.0 * g_big * g_big / d_cb) ** (1.0 / 3.0) * root_n
    return BlockLengths(min(b_sb, b_max), min(b_cb, b_max), m_hat, bw)


def stage4_block_length(ew_returns: FloatArray) -> int:
    """ADR 0016's rule: max(2, round(b_CB)) of |r| of the equal-weight return."""
    return max(2, round(optimal_block_length(np.abs(np.asarray(ew_returns))).circular))
