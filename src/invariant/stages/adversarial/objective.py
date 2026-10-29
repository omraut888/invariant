"""Stage 4's objective: the horizon Sharpe ratio under a state chain (docs/adr/0016).

Method
------
For a day-level chain with transition matrix P, a starting state s0 (the state of the day before
the horizon) and a horizon of H days, the expected share of the horizon spent in each state is

    w = (1/H) sum_{h=1..H} e_{s0}' P^h      (or w = pi for a stationary start).

P is the first-order chain matched to the state's (pi_s, d_s): an approximation (``chain.py``).
Given the strategy's daily P&L mean and sd in each state, ``(mu_k, sd_k)``, SR_H is the
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

from invariant.stages.adversarial.chain import Start, StateChain

__all__ = ["horizon_weights", "mixture_sharpe", "sr_h", "state_moments"]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
TRADING_DAYS = 252


def horizon_weights(chain: StateChain, start: Start, horizon: int) -> FloatArray:
    """Expected share of the ``horizon`` days in (low, high), from ``start``."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if start is Start.STATIONARY:
        return chain.stationary
    p_mat = chain.transition
    row = np.zeros(2)
    row[1 if start is Start.HIGH else 0] = 1.0
    total = np.zeros(2)
    for _ in range(horizon):
        row = row @ p_mat
        total += row
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


def sr_h(mu: FloatArray, sd: FloatArray, chain: StateChain, start: Start, horizon: int) -> float:
    return mixture_sharpe(mu, sd, horizon_weights(chain, start, horizon))
