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
The solution w is invariant to this rescaling. The second-order cone program is passed to
Clarabel (Goulart & Chen 2024) directly: through cvxpy, canonicalisation cost about 3.7 ms per
solve against 0.6 ms for the solve itself, and the study makes millions of solves. The tests
check it against the same problem written in cvxpy.

Assumptions: the centre is an empirical distribution, so sd is its 1/n standard deviation. The
closed form needs c ≥ 0. The ball's guarantee concerns the nominal distribution only; nothing
here accounts for look-ahead, which is the walk-forward driver's job.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import clarabel
import numpy as np
import numpy.typing as npt
import scipy.sparse as sp

__all__ = [
    "MeanStdProblem",
    "Solution",
    "mean_std_risk",
    "robust_mean_std_risk",
    "worst_case_samples",
]

_FloatArray = npt.NDArray[np.float64]
_CASH_TOL: Final = 1e-9
_CASH_GROSS: Final = 1e-6


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
    """``max μᵀw - c‖Lw‖₂ - κ‖w‖₂ s.t. ‖w‖₁ ≤ 1``, solved as a second-order cone program.

    ``c`` is fixed per instance. The variables are ``z = (w, u, s, v)`` and the program is

        min  -μᵀw + c·s + κ·v
        s.t. u - w ≥ 0,  u + w ≥ 0,  1 - Σu ≥ 0,  (s, Lw) ∈ SOC,  (v, w) ∈ SOC,

    in Clarabel's form ``Az + slack = b`` with the slack in the cones. Only L changes between
    solves.
    """

    def __init__(self, n_assets: int, c: float) -> None:
        if c < 0:
            raise ValueError("c must be non-negative")
        self.n_assets = n_assets
        self.c = c
        n = n_assets
        eye, zero = np.eye(n), np.zeros((n, n))
        e_s = np.r_[np.zeros(2 * n), 1.0, 0.0]
        e_v = np.r_[np.zeros(2 * n), 0.0, 1.0]
        self._a = np.vstack(
            [
                np.hstack([eye, -eye, np.zeros((n, 2))]),
                np.hstack([-eye, -eye, np.zeros((n, 2))]),
                np.r_[np.zeros(n), np.ones(n), 0.0, 0.0][None],
                -e_s[None],
                np.zeros((n, 2 * n + 2)),  # -L w, filled per solve
                -e_v[None],
                np.hstack([-eye, zero, np.zeros((n, 2))]),
            ]
        )
        self._l_rows = slice(2 * n + 2, 3 * n + 2)
        self._b = np.r_[np.zeros(2 * n), 1.0, np.zeros(2 * (n + 1))]
        self._p = sp.csc_matrix((2 * n + 2, 2 * n + 2))
        self._cones = [
            clarabel.NonnegativeConeT(2 * n + 1),
            clarabel.SecondOrderConeT(n + 1),
            clarabel.SecondOrderConeT(n + 1),
        ]
        self._settings = clarabel.DefaultSettings()
        self._settings.verbose = False

    def solve(self, mu: _FloatArray, cov: _FloatArray, kappa: float) -> Solution:
        """Solve for forecast ``mu`` and covariance ``cov`` with norm penalty ``kappa``.

        ``kappa = eps·sqrt(1+c²)`` for DRO, 0 for Markowitz and ridge (pass ``cov + λI``).
        The optimum is bang-bang (gross exposure 0 or 1), so a value within tolerance of 0, or an
        interior-point solution with gross exposure ``<= 1e-6``, means cash: the returned w is
        then exactly 0.
        Raises ``RuntimeError`` if the solver does not reach a solved status.
        """
        n = self.n_assets
        if mu.shape != (n,) or cov.shape != (n, n):
            raise ValueError("mu and cov must match n_assets")
        if kappa < 0:
            raise ValueError("kappa must be non-negative")
        scale = 1.0 / float(np.sqrt(np.mean(np.diag(cov))))
        sym = 0.5 * (cov + cov.T) * scale**2
        chol = np.linalg.cholesky(sym + 1e-14 * np.eye(n)).T  # upper: LᵀL = Σ
        a = self._a.copy()
        a[self._l_rows, :n] = -chol
        q = np.r_[-mu * scale, np.zeros(n), self.c, float(kappa) * scale]
        solver = clarabel.DefaultSolver(
            self._p, q, sp.csc_matrix(a), self._b, self._cones, self._settings
        )
        sol = solver.solve()
        status = str(sol.status)
        if sol.status not in (clarabel.SolverStatus.Solved, clarabel.SolverStatus.AlmostSolved):
            raise RuntimeError(f"portfolio solve failed: {status}")
        value_scaled = -float(sol.obj_val)
        w = np.asarray(sol.x[:n], dtype=np.float64)
        if value_scaled <= _CASH_TOL or float(np.abs(w).sum()) <= _CASH_GROSS:
            return Solution(w=np.zeros(n), value=0.0, status=status)
        w = w / max(1.0, float(np.abs(w).sum()))  # strip solver overshoot of the L1 bound
        return Solution(w=w, value=value_scaled / scale, status=status)
