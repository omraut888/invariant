r"""Worst-case mean-std risk over a 2-Wasserstein ball, and the portfolio problem it induces
(docs/adr/0013).

Method
------
For a portfolio w and a nominal distribution of returns with mean μ̂ and covariance Σ̂ (an
empirical distribution: forecast plus training residuals, Kannan, Bayraksan & Luedtke 2024),
the mean-std risk of the loss -wᵀR is ``risk(w) = -μ̂ᵀw + c·sqrt(wᵀΣ̂w)`` with c ≥ 0. Over the
2-Wasserstein ball of radius ε (transport cost ``‖·‖₂²``) around the **empirical** centre, its
worst case is exactly

    sup_Q risk_Q(w) = -μ̂ᵀw + c·sqrt(wᵀΣ̂w) + ε·sqrt(1 + c²)·‖w‖₂ .

This is Nguyen, Shafiee, Filipović & Kuhn (arXiv:2112.09959, Theorem 5, eq. 8) for a Gelbrich
ball. For an empirical centre it is attained by moving every sample point along w
(``worst_case_samples``; derivation in docs/adr/0013). The robust portfolio is then

    max over w of   μ̂ᵀw - c·‖Lw‖₂ - κ‖w‖₂      subject to ‖w‖₁ ≤ 1,     κ = ε·sqrt(1 + c²),

with ``LᵀL = Σ̂`` (or ``Σ̂ + λI`` for ridge). The objective is positively homogeneous of degree
one, so the optimum is either w = 0 (cash) or on the boundary ‖w‖₁ = 1. **c is a Sharpe hurdle:**
the portfolio trades only when some fully invested direction earns more than c*sd + κ‖w‖₂ per
unit of gross exposure.

Numerics: the problem is solved in rescaled units (returns divided by the mean daily
volatility). Raw daily units, with covariances around 1e-4, made SCS inaccurate (docs/adr/0013).
The solution w is invariant to this rescaling.

Assumptions: the centre is an empirical distribution, so sd is its 1/n standard deviation. The
closed form needs c ≥ 0. The ball's guarantee concerns the nominal distribution only; nothing
here accounts for look-ahead, which is the walk-forward driver's job.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import cvxpy as cp
import numpy as np
import numpy.typing as npt

__all__ = [
    "MeanStdProblem",
    "Solution",
    "mean_std_risk",
    "robust_mean_std_risk",
    "worst_case_samples",
]

_FloatArray = npt.NDArray[np.float64]
_CASH_TOL: Final = 1e-9


def mean_std_risk(samples: _FloatArray, w: _FloatArray, c: float) -> float:
    """``-mean(wᵀR) + c·std(wᵀR)`` of an empirical distribution (1/n moments)."""
    y = samples @ w
    return float(-y.mean() + c * y.std())


def robust_mean_std_risk(
    mu: _FloatArray, cov: _FloatArray, w: _FloatArray, c: float, eps: float
) -> float:
    """Closed-form worst case over the 2-Wasserstein ball of radius ``eps`` (module doc)."""
    if c < 0 or eps < 0:
        raise ValueError("c and eps must be non-negative")
    sd = float(np.sqrt(max(float(w @ cov @ w), 0.0)))
    return float(-mu @ w + c * sd + eps * np.sqrt(1.0 + c * c) * np.linalg.norm(w))


def worst_case_samples(samples: _FloatArray, w: _FloatArray, c: float, eps: float) -> _FloatArray:
    """The empirical distribution in the ball that attains ``robust_mean_std_risk``.

    Every point moves along w by ``d_i w / ‖w‖²`` with ``d = ε‖w‖(a + b·z_i)``, where z is
    the standardised projection, ``a = -1/sqrt(1+c²)`` and ``b = c/sqrt(1+c²)``. The transport
    cost ``(1/n) Σ ‖Δ_i‖²`` equals ε² exactly. A degenerate projection (zero std) moves only
    the mean.
    """
    y = samples @ w
    s = float(y.std())
    norm_w = float(np.linalg.norm(w))
    if norm_w == 0.0:
        return samples.copy()
    k = np.sqrt(1.0 + c * c)
    z = (y - y.mean()) / s if s > 0 else np.zeros_like(y)
    a, b = (-1.0 / k, c / k) if s > 0 else (-1.0, 0.0)
    d = eps * norm_w * (a + b * z)
    return np.asarray(samples + np.outer(d, w) / (norm_w**2), dtype=np.float64)


@dataclass(frozen=True)
class Solution:
    """A solved portfolio. ``value`` is the robust objective at w in the caller's units."""

    w: _FloatArray
    value: float
    status: str

    @property
    def invested(self) -> bool:
        return bool(np.abs(self.w).sum() > 0.0)


class MeanStdProblem:
    """``max μᵀw - c‖Lw‖₂ - κ‖w‖₂ s.t. ‖w‖₁ ≤ 1``, compiled once and re-solved with new data.

    ``c`` is fixed per instance. μ, L and κ are cvxpy Parameters, so a re-solve does not
    recompile (DPP). The default solver is Clarabel.
    """

    def __init__(self, n_assets: int, c: float, *, solver: str = cp.CLARABEL) -> None:
        if c < 0:
            raise ValueError("c must be non-negative")
        self.n_assets = n_assets
        self.c = c
        self._solver = solver
        self._mu = cp.Parameter(n_assets)
        self._chol = cp.Parameter((n_assets, n_assets))
        self._kappa = cp.Parameter(nonneg=True)
        self._w = cp.Variable(n_assets)
        objective = (
            self._mu @ self._w
            - c * cp.norm(self._chol @ self._w, 2)
            - self._kappa * cp.norm(self._w, 2)
        )
        self._problem = cp.Problem(cp.Maximize(objective), [cp.norm(self._w, 1) <= 1])

    def solve(self, mu: _FloatArray, cov: _FloatArray, kappa: float) -> Solution:
        """Solve for forecast ``mu`` and covariance ``cov`` with norm penalty ``kappa``.

        ``kappa = eps·sqrt(1+c²)`` for DRO, 0 for Markowitz and ridge (pass ``cov + λI``).
        A value within tolerance of 0 means cash: the returned w is then exactly 0.
        Raises ``RuntimeError`` if the solver does not reach an optimal status.
        """
        if mu.shape != (self.n_assets,) or cov.shape != (self.n_assets, self.n_assets):
            raise ValueError("mu and cov must match n_assets")
        scale = 1.0 / float(np.sqrt(np.mean(np.diag(cov))))
        sym = 0.5 * (cov + cov.T) * scale**2
        chol = np.linalg.cholesky(sym + 1e-14 * np.eye(self.n_assets)).T  # upper: LᵀL = Σ
        self._mu.value = mu * scale
        self._chol.value = chol
        self._kappa.value = float(kappa) * scale
        self._problem.solve(solver=self._solver)
        status = str(self._problem.status)
        if status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or self._w.value is None:
            raise RuntimeError(f"portfolio solve failed: {status}")
        w = np.asarray(self._w.value, dtype=np.float64)
        value_scaled = float(self._problem.value)
        if value_scaled <= _CASH_TOL:
            return Solution(w=np.zeros(self.n_assets), value=0.0, status=status)
        w = w / max(1.0, float(np.abs(w).sum()))  # strip solver overshoot of the L1 bound
        return Solution(w=w, value=value_scaled / scale, status=status)
