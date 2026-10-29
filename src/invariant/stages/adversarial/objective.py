"""Stage 4's objective: the horizon Sharpe ratio under a state chain (docs/adr/0016).

Method
------
For a chain over block types with transition matrix P, a starting type s0, blocks of L days and
a horizon of H days (J = ceil(H / L) slots, slot j lasting l_j days: L, except the last), the
expected share of the horizon's days in each type is

    w = sum_{j=1..J} (l_j / H) e_{s0}' P^j      (or w = pi for a stationary start).

With L = 1 this is the day-level chain. Given the strategy's daily P&L mean and sd within each
type, ``(mu_k, sd_k)``, SR_H is the
annualised Sharpe ratio of the w-mixture: the mean is ``w' mu``, the variance is
``w'(sd^2 + mu^2) - (w' mu)^2``, and the ratio is scaled by sqrt(252). It is the Sharpe ratio of
a day drawn at random from the horizon. It depends on the data only through the per-state
moments, so serial dependence enters through the interval (a block bootstrap), not here.

Assumptions: the state-conditional P&L distribution does not depend on the state mix (the
fixed-policy approximation, ADR 0016). This is the expected-horizon (mixture) Sharpe ratio, not
the mean of realised H-day Sharpe ratios; that path distribution is reported separately.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from invariant.stages.adversarial.generator import Start, StateChain

__all__ = ["horizon_weights", "mixture_sharpe", "sr_h", "state_moments"]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
TRADING_DAYS = 252


def horizon_weights(
    chain: StateChain, start: Start, horizon: int, block_length: int = 1
) -> FloatArray:
    """Expected share of the ``horizon`` days in (low, high) types, from ``start``."""
    if horizon < 1 or block_length < 1:
        raise ValueError("horizon and block_length must be >= 1")
    if start is Start.STATIONARY:
        return chain.stationary
    p_mat = chain.transition
    row = np.zeros(2)
    row[1 if start is Start.HIGH else 0] = 1.0
    total = np.zeros(2)
    n_slots = math.ceil(horizon / block_length)
    for j in range(n_slots):
        row = row @ p_mat
        days = min(block_length, horizon - j * block_length)
        total += row * days
    return total / horizon


def mixture_sharpe(mu: FloatArray, sd: FloatArray, w: FloatArray) -> float:
    """Annualised Sharpe ratio of the w-mixture of per-state P&L moments (0 if no variance)."""
    mean = float(w @ mu)
    var = float(w @ (sd * sd + mu * mu)) - mean * mean
    return 0.0 if var <= 0.0 else mean / math.sqrt(var) * math.sqrt(TRADING_DAYS)


def state_moments(pnl: FloatArray, states: IntArray) -> tuple[FloatArray, FloatArray]:
    """Per-state P&L mean and sd (ddof 0, the plug-in moments)."""
    mu = np.array([pnl[states == k].mean() for k in (0, 1)])
    sd = np.array([pnl[states == k].std() for k in (0, 1)])
    return mu, sd


def sr_h(
    mu: FloatArray,
    sd: FloatArray,
    chain: StateChain,
    start: Start,
    horizon: int,
    block_length: int = 1,
) -> float:
    return mixture_sharpe(mu, sd, horizon_weights(chain, start, horizon, block_length))
