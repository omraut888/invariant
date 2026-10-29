"""Markov-switching GARCH regime labeller for Stage 4 (docs/adr/0016).

Method
------
Haas, Mittnik & Paolella (2004), "A new approach to Markov-switching GARCH models", Journal of
Financial Econometrics 2(4). A latent Markov chain ``xi_t`` in {0, .., K-1} with transition
matrix P selects which of K GARCH(1,1) variance processes drives the observed shock:

    r_t = mu + eps_t,      eps_t | xi_t = k  ~  sqrt(h_{k,t}) z_t,   z ~ unit-variance t(nu)
    h_{k,t} = omega_k + alpha_k eps_{t-1}^2 + beta_k h_{k,t-1}

Every regime's recursion runs on the same observed ``eps`` (HMP's key device), so ``h_{k,t}`` is
a deterministic function of the data and the model is a hidden Markov model with time-varying
emission densities. The likelihood is exact via the Hamilton (1989) filter, with no path
dependence (unlike Gray 1996, which must collapse the variance across regimes). The chain starts
from its stationary distribution, and ``h_{k,0}`` from regime k's unconditional variance.

Estimation is by maximum likelihood (L-BFGS-B on an unconstrained reparameterisation, with
``alpha_k + beta_k < 1`` and ``nu > 2.05`` built in) from a fixed set of starting values, with no
randomness. The best start wins. Regimes are ordered by unconditional variance
``omega_k / (1 - alpha_k - beta_k)``, so the last regime is stress.

**Number of regimes.** K = 1 (a plain GARCH-t) and K = 2 are both fitted, and BIC chooses.
- BIC is a heuristic here: under K = 1 the K = 2 model's extra parameters are unidentified, so
  the usual likelihood-ratio asymptotics fail (Hansen 1992). Its behaviour is therefore measured
  on the benchmark (ADR 0016, L3) rather than assumed.
- With K = 1 every day is labelled calm.

Stage 4 applies the labeller to the panel's equal-weight daily return, and hard-labels each day
by smoothed stress probability > 0.5. Smoothing is two-sided (it uses the whole sample). That is
allowed because labels never feed a trading decision: they only partition observed days for the
regime bootstrap, after Stage 3's walk-forward is fixed.

Assumptions and limitations:
- the mean is common to both regimes (a switching mean would make eps regime-dependent, and with
  it HMP's recursions);
- one tail index nu for both regimes;
- symmetric GARCH (no leverage);
- the chain is exogenous to returns.

``gaussian=True, garch=False`` gives the Markov-switching-variance model of Hamilton (1989)
(constant ``h_k = omega_k``), the restricted case that is cross-checked against statsmodels.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Final, TypeVar, cast

import numba
import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field
from scipy import optimize

__all__ = [
    "MarkovSwitchingGarchFit",
    "RegimeSelection",
    "fit_ms_garch",
    "label_regimes",
    "loglik",
    "select_regimes",
    "smoothed_probabilities",
]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
_F = TypeVar("_F", bound=Callable[..., object])
_jit = cast(Callable[[_F], _F], numba.njit(cache=True))

_NU_MIN: Final = 2.05
_PERSIST_MAX: Final = 0.9995


# ---------------------------------------------------------------------------------------------
# Kernels (numba)
# ---------------------------------------------------------------------------------------------


@_jit
def _log_emissions(
    eps: FloatArray,
    omega: FloatArray,
    alpha: FloatArray,
    beta: FloatArray,
    nu: float,
    gaussian: bool,
) -> FloatArray:
    """``log f_k(eps_t | h_{k,t})``, shape (T, K); ``h_{k,0}`` = unconditional variance."""
    n, k_reg = eps.shape[0], omega.shape[0]
    out = np.empty((n, k_reg))
    if gaussian:
        const = -0.5 * math.log(2.0 * math.pi)
    else:
        const = (
            math.lgamma((nu + 1.0) / 2.0)
            - math.lgamma(nu / 2.0)
            - 0.5 * math.log(math.pi * (nu - 2.0))
        )
    for k in range(k_reg):
        h = omega[k] / (1.0 - alpha[k] - beta[k])
        for t in range(n):
            if t > 0:
                h = omega[k] + alpha[k] * eps[t - 1] * eps[t - 1] + beta[k] * h
            e2 = eps[t] * eps[t] / h
            if gaussian:
                out[t, k] = const - 0.5 * math.log(h) - 0.5 * e2
            else:
                out[t, k] = (
                    const - 0.5 * math.log(h) - 0.5 * (nu + 1.0) * math.log1p(e2 / (nu - 2.0))
                )
    return out


@_jit
def _filter(log_dens: FloatArray, p_mat: FloatArray, init: FloatArray) -> tuple[FloatArray, float]:
    """Hamilton filter: ``(filtered P(xi_t | r_1..t), loglik)``, scaled per step for stability."""
    n, k_reg = log_dens.shape
    filt = np.empty((n, k_reg))
    pred = init.copy()
    ll = 0.0
    for t in range(n):
        m = log_dens[t].max()
        a = pred * np.exp(log_dens[t] - m)
        s = a.sum()
        ll += math.log(s) + m
        filt[t] = a / s
        pred = filt[t] @ p_mat
    return filt, ll


@_jit
def _smooth(filt: FloatArray, p_mat: FloatArray) -> FloatArray:
    """Kim (1994) smoother: ``P(xi_t | r_1..T)`` from the filtered probabilities."""
    n, k_reg = filt.shape
    out = np.empty((n, k_reg))
    out[-1] = filt[-1]
    for t in range(n - 2, -1, -1):
        pred = filt[t] @ p_mat
        ratio = np.where(pred > 0.0, out[t + 1] / pred, 0.0)
        out[t] = filt[t] * (p_mat @ ratio)
        out[t] /= out[t].sum()
    return out


# ---------------------------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------------------------


class MarkovSwitchingGarchFit(BaseModel):
    """A fitted MS-GARCH(1,1)-t. Regimes ordered by unconditional variance (last = stress)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    n_regimes: int = Field(ge=1, le=2)
    mu: float
    omega: tuple[float, ...]
    alpha: tuple[float, ...]
    beta: tuple[float, ...]
    nu: float  # math.inf for the Gaussian model
    transition: tuple[tuple[float, ...], ...]
    loglik: float
    n_params: int
    n_obs: int
    converged: bool

    @property
    def bic(self) -> float:
        return -2.0 * self.loglik + self.n_params * math.log(self.n_obs)

    @property
    def unconditional_variance(self) -> tuple[float, ...]:
        return tuple(
            w / (1.0 - a - b) for w, a, b in zip(self.omega, self.alpha, self.beta, strict=True)
        )

    @property
    def stationary(self) -> tuple[float, ...]:
        return tuple(float(v) for v in _stationary(np.array(self.transition)))

    @property
    def stress_share(self) -> float:
        return self.stationary[-1] if self.n_regimes == 2 else 0.0

    @property
    def stress_duration(self) -> float:
        """Expected stress spell ``1 / P(stress -> calm)``; 0 with one regime."""
        if self.n_regimes == 1:
            return 0.0
        p = self.transition[1][0]
        return math.inf if p == 0.0 else 1.0 / p


class RegimeSelection(BaseModel):
    """Both fits and the BIC choice."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    one: MarkovSwitchingGarchFit
    two: MarkovSwitchingGarchFit

    @property
    def chosen(self) -> MarkovSwitchingGarchFit:
        return self.two if self.two.bic < self.one.bic else self.one


def _stationary(p_mat: FloatArray) -> FloatArray:
    k = p_mat.shape[0]
    if k == 1:
        return np.ones(1)
    a = np.vstack([p_mat.T - np.eye(k), np.ones(k)])
    b = np.zeros(k + 1)
    b[-1] = 1.0
    pi = np.linalg.lstsq(a, b, rcond=None)[0]
    return np.asarray(np.clip(pi, 0.0, 1.0) / np.clip(pi, 0.0, 1.0).sum())


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


class _Spec:
    """Unconstrained vector <-> parameters for K regimes, scaled by the sample variance.

    Per regime: log(v_k / var), logit(persistence / _PERSIST_MAX) and logit(alpha / persistence)
    (both GARCH entries only when ``garch``). Then log(nu - _NU_MIN) unless ``gaussian``, and with
    K = 2 logit(p_00) and logit(p_11). ``mu`` comes first, in units of the sample sd.
    """

    def __init__(self, k: int, *, gaussian: bool, garch: bool, var: float, sd: float) -> None:
        self.k, self.gaussian, self.garch, self.var, self.sd = k, gaussian, garch, var, sd

    @property
    def size(self) -> int:
        return (
            1
            + self.k * (3 if self.garch else 1)
            + (0 if self.gaussian else 1)
            + (2 if self.k == 2 else 0)
        )

    def unpack(
        self, x: FloatArray
    ) -> tuple[float, FloatArray, FloatArray, FloatArray, float, FloatArray]:
        i = 1
        mu = float(x[0]) * self.sd
        omega, alpha, beta = np.empty(self.k), np.zeros(self.k), np.zeros(self.k)
        for k in range(self.k):
            v = math.exp(float(x[i])) * self.var
            i += 1
            if self.garch:
                pers = _PERSIST_MAX * _sigmoid(float(x[i]))
                alpha[k] = pers * _sigmoid(float(x[i + 1]))
                beta[k] = pers - alpha[k]
                i += 2
            omega[k] = v * (1.0 - alpha[k] - beta[k])
        nu = math.inf
        if not self.gaussian:
            nu = _NU_MIN + math.exp(float(x[i]))
            i += 1
        p_mat = np.ones((1, 1))
        if self.k == 2:
            p00, p11 = _sigmoid(float(x[i])), _sigmoid(float(x[i + 1]))
            p_mat = np.array([[p00, 1.0 - p00], [1.0 - p11, p11]])
        return mu, omega, alpha, beta, nu, p_mat

    def pack(
        self,
        mu: float,
        v: list[float],
        pers: list[float],
        alpha_share: list[float],
        nu: float,
        stay: tuple[float, float],
    ) -> FloatArray:
        x = [mu / self.sd]
        for k in range(self.k):
            x.append(math.log(v[k] / self.var))
            if self.garch:
                x += [_logit(pers[k] / _PERSIST_MAX), _logit(alpha_share[k])]
        if not self.gaussian:
            x.append(math.log(nu - _NU_MIN))
        if self.k == 2:
            x += [_logit(stay[0]), _logit(stay[1])]
        return np.array(x)


def _neg_loglik(x: FloatArray, r: FloatArray, spec: _Spec) -> float:
    mu, omega, alpha, beta, nu, p_mat = spec.unpack(x)
    log_dens = _log_emissions(
        r - mu, omega, alpha, beta, 0.0 if spec.gaussian else nu, spec.gaussian
    )
    _, ll = _filter(log_dens, p_mat, _stationary(p_mat))
    return -ll if math.isfinite(ll) else 1e300


def _starts(spec: _Spec, r: FloatArray) -> list[FloatArray]:
    mu = float(np.mean(r))
    var = spec.var
    if spec.k == 1:
        return [spec.pack(mu, [var], [0.97], [0.07], 6.0, (0.5, 0.5))]
    return [
        spec.pack(mu, [lo * var, hi * var], [0.95, 0.95], [0.07, 0.07], 6.0, stay)
        for lo, hi in ((0.6, 3.0), (0.4, 2.0))
        for stay in ((0.99, 0.97), (0.995, 0.99))
    ]


def fit_ms_garch(
    r: FloatArray,
    n_regimes: int,
    *,
    gaussian: bool = False,
    garch: bool = True,
) -> MarkovSwitchingGarchFit:
    """Fit a K-regime MS-GARCH(1,1)-t by maximum likelihood (see module docstring)."""
    x = np.ascontiguousarray(r, dtype=np.float64)
    if x.ndim != 1 or len(x) < 100 or not np.all(np.isfinite(x)):
        raise ValueError("r must be a finite 1-d series of at least 100 observations")
    if n_regimes not in (1, 2):
        raise ValueError("n_regimes must be 1 or 2")
    spec = _Spec(n_regimes, gaussian=gaussian, garch=garch, var=float(x.var()), sd=float(x.std()))
    best: optimize.OptimizeResult | None = None
    for x0 in _starts(spec, x):
        res = optimize.minimize(
            _neg_loglik, x0, args=(x, spec), method="L-BFGS-B", options={"maxiter": 2000}
        )
        if best is None or res.fun < best.fun:
            best = res
    assert best is not None
    mu, omega, alpha, beta, nu, p_mat = spec.unpack(best.x)
    order = np.argsort(omega / (1.0 - alpha - beta), kind="stable")
    omega, alpha, beta = omega[order], alpha[order], beta[order]
    p_mat = p_mat[np.ix_(order, order)]
    return MarkovSwitchingGarchFit(
        n_regimes=n_regimes,
        mu=mu,
        omega=tuple(float(v) for v in omega),
        alpha=tuple(float(v) for v in alpha),
        beta=tuple(float(v) for v in beta),
        nu=nu,
        transition=tuple(tuple(float(v) for v in row) for row in p_mat),
        loglik=-float(best.fun),
        n_params=spec.size,
        n_obs=len(x),
        converged=bool(best.success),
    )


def select_regimes(r: FloatArray) -> RegimeSelection:
    """Fit K = 1 and K = 2 (MS-GARCH-t) and let BIC choose (``RegimeSelection.chosen``)."""
    return RegimeSelection(one=fit_ms_garch(r, 1), two=fit_ms_garch(r, 2))


def _filtered(r: FloatArray, fit: MarkovSwitchingGarchFit) -> tuple[FloatArray, float]:
    x = np.ascontiguousarray(r, dtype=np.float64)
    gaussian = math.isinf(fit.nu)
    log_dens = _log_emissions(
        x - fit.mu,
        np.array(fit.omega),
        np.array(fit.alpha),
        np.array(fit.beta),
        0.0 if gaussian else fit.nu,
        gaussian,
    )
    p_mat = np.array(fit.transition)
    return _filter(log_dens, p_mat, _stationary(p_mat))


def loglik(r: FloatArray, fit: MarkovSwitchingGarchFit) -> float:
    """Exact log-likelihood of ``r`` at ``fit``'s parameters."""
    return _filtered(r, fit)[1]


def smoothed_probabilities(r: FloatArray, fit: MarkovSwitchingGarchFit) -> FloatArray:
    """``P(xi_t = k | r_1..r_T)``, shape (T, K)."""
    filt, _ = _filtered(r, fit)
    return _smooth(filt, np.array(fit.transition))


def label_regimes(r: FloatArray, fit: MarkovSwitchingGarchFit) -> IntArray:
    """Hard labels: 1 (stress) where the smoothed stress probability exceeds 0.5; all 0 if K = 1."""
    if fit.n_regimes == 1:
        return np.zeros(len(r), dtype=np.int64)
    return (smoothed_probabilities(r, fit)[:, 1] > 0.5).astype(np.int64)
