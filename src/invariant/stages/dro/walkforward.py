r"""Strictly walk-forward forecasts and portfolios for Stage 3's arms (docs/adr/0013).

Timing
------
A feature dated t is known at the close of t, and ``returns[t]`` is the return from close
t-1 to close t (ADR 0003). A forecast made at t targets ``returns[t+1]``. Every quantity used
at date t comes only from rows <= t:

- **Refit** every ``refit_every`` rows at a date tau, on the ``window`` most recent pairs
  ``(x[s], r[s+1])`` with ``s + 1 <= tau``. For each asset this is OLS on ``[1, x]``, giving
  the coefficients B, the 1/n residual covariance, and a Newey-West HAC covariance of the
  stacked coefficients across assets.
- **Daily**: ``mu_t = B [1, x_t]``. Each arm solves its problem on ``mu_t`` and holds the
  weights from close t to close t+1, earning ``w_t . r[t+1]``.
- **CV** (Ridge-CV, DRO-CV) every ``cv_every`` rows, inside the current training window and
  walk-forward only. The model is fitted on the first ``window - cv_valid`` pairs, and each grid
  value is then run daily over the last ``cv_valid`` pairs. The score is the realised mean-std
  risk ``-mean + c std`` of those validation returns, the out-of-sample cost of Kannan,
  Bayraksan & Luedtke (2024, sec. 6) with blocked folds in place of their random K-fold. The
  lowest score wins, and ties go to the smallest grid value. The choice holds until the next CV
  date.

Arms
----
- ``equal_weight``: ``w = 1/N``, no forecast.
- ``markowitz``: kappa = 0 on the residual covariance.
- ``ridge_cv``: covariance ``S + lam (tr S / N) I``, with lam chosen by CV.
- ``dro_cv``: the 2-Wasserstein penalty ``kappa = eps sqrt(1 + c^2)``, with
  ``eps = e sqrt(tr S / N)`` and e chosen by CV.
- ``dro_fc``: the same penalty, with ``eps_t`` the (1 - alpha) quantile of ``||mu_hat_t -
  mu_t||_2`` under ``N(0, V_t)``. ``V_t`` is the HAC forecast covariance, and the quantile of
  the weighted chi-square is taken from fixed seeded draws. With that eps, the worst-case mean
  ``mu_hat' w - eps ||w||_2`` (Blanchet, Chen & Zhou 2018, Prop. 1) is below the true mean for
  every w with probability 1 - alpha. The arm calibrates the mean channel only.

Assumptions and limitations:
- The forecast covariance assumes the coefficient estimates are approximately normal.
- The Newey-West rule-of-thumb bandwidth does not capture the omitted Z's persistence
  (phi .99, ADR 0003), so the radius may undercover. Its coverage is reported against the
  oracle.
- Position drift between rebalances and trading costs are ignored here; costs are applied in
  the study's sensitivity.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
import numpy.typing as npt

from invariant.benchmark.diagnostics import newey_west_lags
from invariant.core.run_hash import RngFactory
from invariant.stages.dro.robust import MeanStdProblem

__all__ = [
    "Arm",
    "ArmPath",
    "LinearFit",
    "WalkForwardConfig",
    "fc_radius",
    "fit_linear",
    "run_walk_forward",
    "solve_path",
]

_FloatArray = npt.NDArray[np.float64]


class Arm(StrEnum):
    EQUAL_WEIGHT = "equal_weight"
    MARKOWITZ = "markowitz"
    DRO_FC = "dro_fc"
    RIDGE_CV = "ridge_cv"
    DRO_CV = "dro_cv"


@dataclass(frozen=True)
class WalkForwardConfig:
    c: float
    window: int = 1260
    refit_every: int = 21
    cv_every: int = 252
    cv_valid: int = 252
    fc_alpha: float = 0.05
    fc_draws: int = 4000
    ridge_grid: tuple[float, ...] = (0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
    eps_grid: tuple[float, ...] = (0.0, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2)

    def __post_init__(self) -> None:
        if self.c < 0:
            raise ValueError("c must be non-negative")
        if not 0 < self.cv_valid < self.window:
            raise ValueError("cv_valid must be in (0, window)")
        if self.refit_every < 1 or self.cv_every < 1:
            raise ValueError("refit_every and cv_every must be >= 1")


@dataclass(frozen=True)
class LinearFit:
    """Per-asset OLS of ``r[s+1]`` on ``[1, x[s]]``. ``coef[i]`` is asset i's vector, and
    ``coef_cov[i, a, j, b]`` the HAC covariance of ``coef[i, a]`` and ``coef[j, b]``."""

    coef: _FloatArray  # (N, K+1)
    resid_cov: _FloatArray  # (N, N), 1/n
    coef_cov: _FloatArray  # (N, K+1, N, K+1)

    def forecast(self, x_t: _FloatArray) -> _FloatArray:
        return np.asarray(self.coef @ np.r_[1.0, x_t], dtype=np.float64)

    def forecast_cov(self, x_t: _FloatArray) -> _FloatArray:
        d = np.r_[1.0, x_t]
        return np.asarray(np.einsum("iajb,a,b->ij", self.coef_cov, d, d), dtype=np.float64)


def fit_linear(x: _FloatArray, y: _FloatArray, *, hac_lags: int | None = None) -> LinearFit:
    """OLS on ``[1, x]`` for every column of y, with a Newey-West (Bartlett) HAC covariance of
    the stacked coefficients. ``hac_lags=None`` uses the rule-of-thumb ``newey_west_lags``."""
    n, n_assets = y.shape
    d = np.column_stack([np.ones(n), x])
    k1 = d.shape[1]
    xtx_inv = np.linalg.inv(d.T @ d)
    coef = (xtx_inv @ d.T @ y).T  # (N, K+1)
    e = y - d @ coef.T
    g = (e[:, :, None] * d[:, None, :]).reshape(n, n_assets * k1)  # score of every coefficient
    lags = newey_west_lags(n) if hac_lags is None else hac_lags
    omega = g.T @ g
    for lag in range(1, lags + 1):
        gamma = g[lag:].T @ g[:-lag]
        omega += (1.0 - lag / (lags + 1.0)) * (gamma + gamma.T)
    a = np.kron(np.eye(n_assets), xtx_inv)
    coef_cov = (a @ omega @ a).reshape(n_assets, k1, n_assets, k1)
    return LinearFit(coef=coef, resid_cov=e.T @ e / n, coef_cov=coef_cov)


def fc_radius(v: _FloatArray, draws: _FloatArray, alpha: float) -> float:
    """``sqrt`` of the (1 - alpha) quantile of ``||u||^2`` with ``u ~ N(0, v)``, from standard
    normal ``draws`` (D, N): ``||u||^2 = sum_k lambda_k z_k^2`` over v's eigenvalues."""
    lam = np.clip(np.linalg.eigvalsh(0.5 * (v + v.T)), 0.0, None)
    return float(math.sqrt(np.quantile((draws**2) @ lam, 1.0 - alpha)))


@dataclass
class ArmPath:
    """One arm's daily out-of-sample path. Row k is test date ``dates[k]``: ``weights[k]`` is
    held over ``(dates[k], dates[k] + 1]`` and earns ``returns[k]``."""

    arm: Arm
    dates: npt.NDArray[np.int64]
    weights: _FloatArray
    returns: _FloatArray
    mu_hat: _FloatArray | None = None
    radius: _FloatArray | None = None
    chosen: list[float] = field(default_factory=list)
    """CV arms: the grid value chosen at each CV date."""


def _scale(cov: _FloatArray) -> float:
    return float(math.sqrt(np.trace(cov) / cov.shape[0]))


def _weights(
    arm: Arm,
    prob: MeanStdProblem,
    fit: LinearFit,
    x_t: _FloatArray,
    c: float,
    grid_value: float,
    radius: float | None,
) -> _FloatArray:
    n = fit.coef.shape[0]
    if arm is Arm.EQUAL_WEIGHT:
        return np.full(n, 1.0 / n)
    mu, cov = fit.forecast(x_t), fit.resid_cov
    k = math.sqrt(1.0 + c * c)
    if arm is Arm.MARKOWITZ:
        return prob.solve(mu, cov, 0.0).w
    if arm is Arm.RIDGE_CV:
        return prob.solve(mu, cov + grid_value * _scale(cov) ** 2 * np.eye(n), 0.0).w
    if arm is Arm.DRO_CV:
        return prob.solve(mu, cov, grid_value * _scale(cov) * k).w
    assert radius is not None
    return prob.solve(mu, cov, radius * k).w


def _cv_choice(
    arm: Arm,
    prob: MeanStdProblem,
    x: _FloatArray,
    r: _FloatArray,
    tau: int,
    cfg: WalkForwardConfig,
) -> float:
    """Walk-forward CV inside the window ending at tau (module docstring)."""
    grid = cfg.ridge_grid if arm is Arm.RIDGE_CV else cfg.eps_grid
    lo, mid = tau - cfg.window, tau - cfg.cv_valid
    sub = fit_linear(x[lo:mid], r[lo + 1 : mid + 1])
    scores = []
    for g in grid:
        rets = np.array(
            [_weights(arm, prob, sub, x[s], cfg.c, g, None) @ r[s + 1] for s in range(mid, tau)]
        )
        scores.append(float(-rets.mean() + cfg.c * rets.std()))
    return float(grid[int(np.argmin(scores))])  # first minimum: the smallest grid value


def run_walk_forward(
    x: _FloatArray,
    r: _FloatArray,
    arms: Sequence[Arm],
    cfg: WalkForwardConfig,
    rng: RngFactory,
) -> dict[Arm, ArmPath]:
    """Every arm's out-of-sample path on features ``x`` (T, K) and returns ``r`` (T, N).

    Test dates run from ``window`` to ``T - 2``. Deterministic given ``rng``, which supplies the
    radius draws (stream ``fc_draws``).
    """
    n_t, n_assets = r.shape
    if x.shape[0] != n_t:
        raise ValueError("x and r must have the same rows")
    if n_t < cfg.window + 2:
        raise ValueError("not enough rows for one out-of-sample day")
    prob = MeanStdProblem(n_assets, cfg.c)
    draws = rng.generator("fc_draws").standard_normal((cfg.fc_draws, n_assets))
    dates = np.arange(cfg.window, n_t - 1, dtype=np.int64)
    paths = {
        a: ArmPath(
            a,
            dates,
            np.zeros((len(dates), n_assets)),
            np.zeros(len(dates)),
            mu_hat=np.zeros((len(dates), n_assets)) if a is Arm.DRO_FC else None,
            radius=np.zeros(len(dates)) if a is Arm.DRO_FC else None,
        )
        for a in arms
    }
    choice = {a: 0.0 for a in arms}
    fit: LinearFit | None = None
    for k, t in enumerate(dates):
        offset = int(t) - cfg.window
        if offset % cfg.refit_every == 0:
            fit = fit_linear(x[t - cfg.window : t], r[t - cfg.window + 1 : t + 1])
        if offset % cfg.cv_every == 0:
            for a in arms:
                if a in (Arm.RIDGE_CV, Arm.DRO_CV):
                    choice[a] = _cv_choice(a, prob, x, r, int(t), cfg)
                    paths[a].chosen.append(choice[a])
        assert fit is not None
        for a in arms:
            radius = None
            if a is Arm.DRO_FC:
                radius = fc_radius(fit.forecast_cov(x[t]), draws, cfg.fc_alpha)
                p = paths[a]
                assert p.mu_hat is not None and p.radius is not None
                p.mu_hat[k] = fit.forecast(x[t])
                p.radius[k] = radius
            w = _weights(a, prob, fit, x[t], cfg.c, choice[a], radius)
            paths[a].weights[k] = w
            paths[a].returns[k] = float(w @ r[t + 1])
    return paths


def solve_path(
    mean: _FloatArray, cov: _FloatArray, r: _FloatArray, dates: npt.NDArray[np.int64], c: float
) -> tuple[_FloatArray, _FloatArray]:
    """Weights and realised returns of the problem solved with *given* forecasts (an oracle):
    ``mean[t]`` and ``cov[t]`` at each date t, kappa = 0."""
    prob = MeanStdProblem(r.shape[1], c)
    w = np.array([prob.solve(mean[t], cov[t], 0.0).w for t in dates])
    rets = np.einsum("tn,tn->t", w, r[dates + 1])
    return w, np.asarray(rets, dtype=np.float64)
