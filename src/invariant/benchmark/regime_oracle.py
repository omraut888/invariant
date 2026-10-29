"""Population oracle for calibrating the v4 regime benchmark (ADR 0015).

Method
------
For one member signal S, the oracle forecasts ``R_{t+1}`` by its population linear projection
on ``S_t``: ``m(S) = a + b S`` with ``b = Cov(S_t, R_{t+1}) / Var(S)``. The slope is the
regime-averaged one, ``E[c] beta`` for S_regime, because a linear forecaster that does not see
the regime can only learn the average (Stage 3's Ridge-CV estimates this slope). The weights
solve Stage 3's mean-std problem, ``max_w m(S)'w - c sqrt(w' Sigma_e w)`` subject to
``||w||_1 <= 1`` (``MeanStdProblem``, ADR 0013), with ``Sigma_e`` the projection residual
covariance. Since ``m`` depends on the scalar S only, ``w(S)`` is solved once on a grid and
looked up by nearest grid point.

The projection moments are estimated on a long training path and the Sharpe ratios measured on
a separate long test path, so "population" here means Monte Carlo with negligible error at the
path lengths used (10^6 days); there is no closed form for regime-conditional Sharpe of a
mean-std portfolio. The per-regime Sharpe is over days whose *return date* is in that regime,
matching ``R_t``'s dependence on ``xi_t``.

Assumptions: the oracle knows the population projection, not the regime, so it is the
best a regime-blind linear strategy can do; Stage 3 does worse (estimation error). It is used
only to choose the benchmark's parameters, never to evaluate a stage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from invariant.benchmark.config import SyntheticMarketConfig
from invariant.benchmark.synthetic import REGIME_NODE, generate_market
from invariant.stages.adversarial.state import volatility_state
from invariant.stages.dro.robust import MeanStdProblem

__all__ = ["OracleRegimeSharpe", "OracleStateSharpe", "oracle_regime_sharpe", "oracle_state_sharpe"]

STAGE3_WINDOW = 1260
"""First decision date of Stage 3's walk-forward; the volatility state is defined on dates from
here on, as Stage 4 defines it."""

FloatArray = npt.NDArray[np.float64]

_GRID = np.linspace(-6.0, 6.0, 401)  # step 0.03 in units of S's (unit) sd


@dataclass(frozen=True)
class OracleRegimeSharpe:
    """Annualised oracle Sharpe ratios by return-date regime, and 1/N's on the same days."""

    member: str
    calm: float
    stress: float
    stationary: float
    equal_weight_calm: float
    equal_weight_stress: float
    equal_weight_stationary: float
    stress_share_test: float


def _sharpe(x: FloatArray) -> float:
    sd = float(x.std(ddof=1))
    return float(x.mean() / sd * math.sqrt(252.0)) if sd > 0 else 0.0


def _projection(s: FloatArray, r: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    """``(a, b, Sigma_e)`` of ``r_{t+1} = a + b s_t + e``, per asset, on aligned arrays."""
    x = s - s.mean()
    b = (x @ (r - r.mean(axis=0))) / (x @ x)
    a = r.mean(axis=0) - b * s.mean()
    resid = r - a - np.outer(s, b)
    return a, b, np.cov(resid, rowvar=False)


def _oracle_weights(
    config: SyntheticMarketConfig, member: str, train_seed: int, c: float
) -> FloatArray:
    """``w(S)`` on ``_GRID``, from the population projection estimated on a training path."""
    train = generate_market(config, train_seed)
    a, b, cov = _projection(train.features[member].to_numpy()[:-1], train.returns.to_numpy()[1:])
    problem = MeanStdProblem(config.n_assets, c)
    return np.array([problem.solve(a + b * g, cov, 0.0).w for g in _GRID])


def _lookup(weights: FloatArray, s: FloatArray) -> FloatArray:
    idx = np.clip(np.rint((s - _GRID[0]) / (_GRID[1] - _GRID[0])).astype(int), 0, len(_GRID) - 1)
    return np.asarray(weights[idx], dtype=np.float64)


@dataclass(frozen=True)
class OracleStateSharpe:
    """Annualised oracle and 1/N Sharpe ratios in Stage 4's observable volatility state."""

    member: str
    high: float
    low: float
    equal_weight_high: float
    equal_weight_low: float


def oracle_state_sharpe(
    config: SyntheticMarketConfig,
    member: str,
    *,
    train_seed: int,
    test_seed: int,
    c: float,
    quantile: float = 0.70,
) -> OracleStateSharpe:
    """Oracle single-signal Sharpe by Stage 4's volatility state (vol21, ``quantile``).

    Decision dates ``t >= STAGE3_WINDOW`` of the test path; the P&L on ``t + 1`` is labelled with
    the state of day t, exactly as ``stages.adversarial.state.volatility_state`` defines it.
    """
    weights = _oracle_weights(config, member, train_seed, c)
    test = generate_market(config, test_seed)
    s = test.features[member].to_numpy()
    r = test.returns.to_numpy()
    dates = np.arange(STAGE3_WINDOW, len(r) - 1, dtype=np.int64)
    pnl = np.einsum("ij,ij->i", _lookup(weights, s[dates]), r[dates + 1])
    ew = r[dates + 1].mean(axis=1)
    state, _ = volatility_state(r.mean(axis=1), dates, quantile=quantile)
    hi, lo = state == 1, state == 0
    return OracleStateSharpe(
        member=member,
        high=_sharpe(pnl[hi]),
        low=_sharpe(pnl[lo]),
        equal_weight_high=_sharpe(ew[hi]),
        equal_weight_low=_sharpe(ew[lo]),
    )


def oracle_regime_sharpe(
    config: SyntheticMarketConfig,
    member: str,
    *,
    train_seed: int,
    test_seed: int,
    c: float,
) -> OracleRegimeSharpe:
    """Oracle Sharpe of a single-signal mean-std portfolio, by regime (see module docstring)."""
    if config.regime is None:
        raise ValueError("oracle_regime_sharpe needs a config with a regime block")
    train = generate_market(config, train_seed)
    s_tr = train.features[member].to_numpy()[:-1]
    r_tr = train.returns.to_numpy()[1:]
    a, b, cov = _projection(s_tr, r_tr)
    problem = MeanStdProblem(config.n_assets, c)
    weights = np.array([problem.solve(a + b * g, cov, 0.0).w for g in _GRID])

    test = generate_market(config, test_seed)
    s_te = test.features[member].to_numpy()[:-1]
    r_te = test.returns.to_numpy()[1:]
    xi = test.latents[REGIME_NODE].to_numpy()[1:]
    idx = np.clip(np.rint((s_te - _GRID[0]) / (_GRID[1] - _GRID[0])).astype(int), 0, len(_GRID) - 1)
    pnl = np.einsum("ij,ij->i", weights[idx], r_te)
    ew = r_te.mean(axis=1)
    calm, stress = xi == 0, xi == 1
    return OracleRegimeSharpe(
        member=member,
        calm=_sharpe(pnl[calm]),
        stress=_sharpe(pnl[stress]),
        stationary=_sharpe(pnl),
        equal_weight_calm=_sharpe(ew[calm]),
        equal_weight_stress=_sharpe(ew[stress]),
        equal_weight_stationary=_sharpe(ew),
        stress_share_test=float(stress.mean()),
    )
