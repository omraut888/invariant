r"""Half-life of a link's edge: change-point exponential model, profile likelihood, LR interval.

Model
-----
The edge is constant until an onset row ``tau``, then decays exponentially:

    m(i; tau, HL) = 1                     for i < tau
                  = 2^{-(i - tau) / HL}   for i >= tau,     HL in (0, inf]

``HL = inf`` is "no decay", and then ``tau`` is not identified. A window statistic's
expectation is a model amplitude times the *window average* ``s_j`` of ``m`` over its rows.
Using the average, not m at the window midpoint, keeps the model exact when the decay is fast
relative to the window. Three estimators share this machinery:

- ``partial_corr`` (estimator c): Fisher z of the window partial correlation, Gaussian with
  variance ``1/(w - k - 3)``, mean ``zeta0 * s_j``. ``zeta0`` takes either sign. The partial
  correlation is proportional to the coefficient, so m is evaluated at the *coefficient*
  half-life ``2 HL``, and HL is reported on the TE scale (low-SNR relation, below).
- ``gaussian_te`` (estimator b): window TE ``-1/2 log(1 - r^2) = log cosh(z)`` is a function of
  ``|z|``. Its exact likelihood is therefore the folded normal of ``|z|`` with ``mu_j = atanh(
  sqrt(1 - exp(-2 TE0 s_j)))``, and ``TE0 >= 0``. Here m is evaluated at HL directly. The
  folded normal absorbs the estimator's positive floor near TE = 0, so no ad hoc bias
  correction is needed.
- ``ksg`` (estimator a): the KSG value minus its surrogate mean, Gaussian with the surrogate
  SD, mean ``TE0 s_j``, ``TE0 >= 0``, evaluated at HL.

Fit and interval: the profile log-likelihood ``l_p(HL) = max over (tau, amplitude)`` is taken
on a grid of tau (every ``tau_step`` rows) and of HL (log-spaced) plus ``inf``. The MLE is the
grid maximum. The 95% interval is every grid HL, and ``inf``, with ``2 (l_max - l_p(HL)) <=
chi2_1(.95)``, reported as its hull. Results are stored as decay rates ``lambda = ln 2 / HL``
(per row, on the TE scale), so "no decay" is ``lambda = 0`` and every number stays finite.
``lambda_low = 0`` means the interval includes no decay.

Assumptions and limitations
---------------------------
- The model is misspecified for a step change, a partial decay to a non-zero floor, or a
  recovery.
- Low-SNR relation: TE is proportional to the squared coefficient, so the TE half-life is
  half the coefficient half-life. It is exact for (b) and (a) by construction, and an
  approximation for (c). At r0 = .25 the exact TE half-life is 2.3% shorter (docs/adr/0011).
- With ``HL = inf`` under the null, ``tau`` is unidentified (Davies 1987). The chi-square
  calibration of the LR is then not guaranteed. The study measures the false-decay rate this
  produces, and nothing assumes it.
- Windows are treated as independent. They do not overlap, but persistent regressors leave
  some correlation between neighbouring windows.
- Grid discretisation: the HL grid ratio is about 8% per step, and tau moves in steps of
  ``tau_step`` rows.

References: Davies (1987), "Hypothesis testing when a nuisance parameter is present only under
the alternative", Biometrika 74(1); Venzon & Moolgavkar (1988), "A method for computing
profile-likelihood-based confidence intervals", Applied Statistics 37(1).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field
from scipy import stats

from invariant.stages.decay.te import KsgWindows
from invariant.stages.decay.windows import WindowPartialCorr

__all__ = [
    "DecayGrid",
    "Estimator",
    "HalfLifeFit",
    "fit_gaussian_te",
    "fit_ksg",
    "fit_partial_corr",
    "make_grid",
    "window_decay_mean",
]

_FloatArray = npt.NDArray[np.float64]
_IntArray = npt.NDArray[np.int64]
_LN2: Final = math.log(2.0)
_TAU_CHUNK: Final = 16
_GOLDEN: Final = (math.sqrt(5.0) - 1.0) / 2.0


class Estimator(StrEnum):
    KSG = "ksg"
    GAUSSIAN_TE = "gaussian_te"
    PARTIAL_CORR = "partial_corr"


@dataclass(frozen=True)
class DecayGrid:
    """Onset rows and TE-scale half-lives (rows) on which the profile likelihood is taken."""

    taus: _IntArray
    half_lives: _FloatArray

    def __post_init__(self) -> None:
        if len(self.taus) < 1 or len(self.half_lives) < 2:
            raise ValueError("the grid needs at least one onset and two half-lives")
        if np.any(self.half_lives <= 0.0) or np.any(np.diff(self.half_lives) <= 0.0):
            raise ValueError("half-lives must be positive and increasing")


def make_grid(n: int, *, tau_step: int, hl_min: float, hl_max: float, n_hl: int) -> DecayGrid:
    """Onsets ``0, tau_step, ...`` below ``n``. ``n_hl`` log-spaced half-lives in ``[hl_min,
    hl_max]``."""
    if tau_step < 1 or n < 1:
        raise ValueError("tau_step and n must be >= 1")
    return DecayGrid(
        taus=np.arange(0, n, tau_step, dtype=np.int64),
        half_lives=np.geomspace(hl_min, hl_max, n_hl),
    )


def window_decay_mean(bounds: _IntArray, taus: _IntArray, half_lives: _FloatArray) -> _FloatArray:
    """``s[t, h, j]``: mean of ``m(i; taus[t], half_lives[h])`` over rows ``i`` of window j.

    Closed form. Rows before the onset contribute 1 each, and rows ``s..b-1`` at or after it
    contribute a geometric sum ``q^{s - tau} (1 - q^{b - s}) / (1 - q)`` with ``q = 2^{-1/HL}``.
    """
    a = bounds[:, 0].astype(np.float64)[None, None, :]
    b = bounds[:, 1].astype(np.float64)[None, None, :]
    tau = taus.astype(np.float64)[:, None, None]
    q = np.exp2(-1.0 / half_lives)[None, :, None]
    before = np.clip(tau - a, 0.0, b - a)
    s = np.maximum(a, tau)
    count = np.maximum(b - s, 0.0)
    # (1 - q^count) / (1 - q) via expm1 for accuracy when q is close to 1
    log_q = np.log(q)
    geom = np.where(count > 0.0, np.expm1(count * log_q) / np.expm1(log_q), 0.0)
    after = np.exp((s - tau) * log_q) * geom
    return np.asarray((before + after) / (b - a), dtype=np.float64)


class HalfLifeFit(BaseModel):
    """A half-life fit on the TE scale, as decay rates per row (``ln 2 / HL``, 0 = no decay).

    ``lam_low > 0`` iff the 95% profile-LR interval excludes "no decay" (the estimator claims
    decay). ``tau_hat`` is the onset at the MLE, which is meaningless when ``lam_hat == 0``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    estimator: Estimator
    lam_hat: float = Field(ge=0.0)
    lam_low: float = Field(ge=0.0)
    lam_high: float = Field(ge=0.0)
    tau_hat: int
    amplitude_hat: float
    loglik_max: float
    loglik_no_decay: float
    n_windows: int = Field(ge=1)

    @property
    def claims_decay(self) -> bool:
        return self.lam_low > 0.0

    def covers(self, te_half_life: float) -> bool:
        lam = _LN2 / te_half_life
        return self.lam_low <= lam <= self.lam_high


# A profile maps shape arrays (..., K) to the log-likelihood (...) maximised over the
# amplitude, and the maximising amplitude.
_Profile = Callable[[_FloatArray], tuple[_FloatArray, _FloatArray]]


def _fit(
    estimator: Estimator,
    bounds: _IntArray,
    grid: DecayGrid,
    profile: _Profile,
    *,
    shape_scale: float,
) -> HalfLifeFit:
    k = len(bounds)
    ll_inf, amp_inf = profile(np.ones((1, k)))
    best_ll = np.full(len(grid.half_lives), -np.inf)
    best_tau = np.zeros(len(grid.half_lives), dtype=np.int64)
    best_amp = np.zeros(len(grid.half_lives))
    for start in range(0, len(grid.taus), _TAU_CHUNK):
        taus = grid.taus[start : start + _TAU_CHUNK]
        shapes = window_decay_mean(bounds, taus, grid.half_lives * shape_scale)
        ll, amp = profile(shapes)  # (n_tau, n_hl)
        idx = np.argmax(ll, axis=0)
        cand = ll[idx, np.arange(ll.shape[1])]
        better = cand > best_ll
        best_ll = np.where(better, cand, best_ll)
        best_tau = np.where(better, taus[idx], best_tau)
        best_amp = np.where(better, amp[idx, np.arange(ll.shape[1])], best_amp)
    l_inf = float(ll_inf[0])
    l_max = max(float(best_ll.max()), l_inf)
    crit = float(stats.chi2.ppf(0.95, 1))
    inside = 2.0 * (l_max - best_ll) <= crit
    inf_inside = 2.0 * (l_max - l_inf) <= crit
    if l_inf >= float(best_ll.max()):
        lam_hat, tau_hat, amp_hat = 0.0, -1, float(amp_inf[0])
    else:
        h = int(np.argmax(best_ll))
        lam_hat = _LN2 / float(grid.half_lives[h])
        tau_hat, amp_hat = int(best_tau[h]), float(best_amp[h])
    hl_in = grid.half_lives[inside]
    lam_high = _LN2 / float(hl_in.min()) if len(hl_in) else 0.0
    lam_low = 0.0 if inf_inside else _LN2 / float(hl_in.max())
    return HalfLifeFit(
        estimator=estimator,
        lam_hat=lam_hat,
        lam_low=lam_low,
        lam_high=max(lam_high, lam_hat),
        tau_hat=tau_hat,
        amplitude_hat=amp_hat,
        loglik_max=l_max,
        loglik_no_decay=l_inf,
        n_windows=k,
    )


def _gaussian_profile(obs: _FloatArray, var: _FloatArray, *, nonneg: bool) -> _Profile:
    w = 1.0 / var
    syy = float(np.sum(obs * obs * w))

    def profile(shapes: _FloatArray) -> tuple[_FloatArray, _FloatArray]:
        a = np.sum(shapes * shapes * w, axis=-1)
        b = np.sum(shapes * obs * w, axis=-1)
        amp = np.where(a > 0.0, b / np.where(a > 0.0, a, 1.0), 0.0)
        if nonneg:
            amp = np.maximum(amp, 0.0)
        ll = -0.5 * (syy - 2.0 * amp * b + amp * amp * a)
        return ll, amp

    return profile


def fit_partial_corr(wpc: WindowPartialCorr, grid: DecayGrid) -> HalfLifeFit:
    """Estimator (c): Gaussian likelihood of the windows' Fisher z (module docstring)."""
    profile = _gaussian_profile(wpc.fisher_z, wpc.fisher_var, nonneg=False)
    return _fit(Estimator.PARTIAL_CORR, wpc.bounds, grid, profile, shape_scale=2.0)


def fit_ksg(ksg: KsgWindows, grid: DecayGrid) -> HalfLifeFit:
    """Estimator (a): KSG value minus surrogate mean, Gaussian with surrogate SD, TE0 >= 0."""
    sd = np.maximum(ksg.null_sd, 1e-12)
    profile = _gaussian_profile(ksg.value - ksg.null_mean, sd * sd, nonneg=True)
    return _fit(Estimator.KSG, ksg.bounds, grid, profile, shape_scale=1.0)


def _folded_ll(abs_z: _FloatArray, var: _FloatArray, te: _FloatArray) -> _FloatArray:
    """Sum over windows (last axis) of the folded-normal log-density of ``|z|``, mean
    ``atanh(sqrt(1 - exp(-2 TE)))``, constants dropped."""
    mu = np.arctanh(np.sqrt(-np.expm1(-2.0 * np.minimum(te, 15.0))).clip(0.0, 1.0 - 1e-15))
    ll = -((abs_z - mu) ** 2) / (2.0 * var) + np.log1p(np.exp(-2.0 * abs_z * mu / var))
    return np.asarray(np.sum(ll, axis=-1), dtype=np.float64)


def fit_gaussian_te(
    wpc: WindowPartialCorr, grid: DecayGrid, *, te_max: float = 2.0, n_iter: int = 40
) -> HalfLifeFit:
    """Estimator (b): folded-normal likelihood of the windows' Gaussian TE (module docstring).

    TE0 is profiled by a coarse log grid on ``[0, te_max]`` that brackets the maximum, then by
    ``n_iter`` golden-section steps inside the bracket, vectorised over every (tau, HL).
    """
    abs_z, var = np.abs(wpc.fisher_z), wpc.fisher_var
    coarse = np.concatenate([[0.0], np.geomspace(1e-6, te_max, 40)])

    def profile(shapes: _FloatArray) -> tuple[_FloatArray, _FloatArray]:
        lls = np.stack([_folded_ll(abs_z, var, c * shapes) for c in coarse], axis=-1)
        i = np.argmax(lls, axis=-1)
        lo = coarse[np.maximum(i - 1, 0)]
        hi = coarse[np.minimum(i + 1, len(coarse) - 1)]
        for _ in range(n_iter):
            c1 = hi - _GOLDEN * (hi - lo)
            c2 = lo + _GOLDEN * (hi - lo)
            f1 = _folded_ll(abs_z, var, c1[..., None] * shapes)
            f2 = _folded_ll(abs_z, var, c2[..., None] * shapes)
            left = f1 >= f2
            hi = np.where(left, c2, hi)
            lo = np.where(left, lo, c1)
        amp = 0.5 * (lo + hi)
        ll = _folded_ll(abs_z, var, amp[..., None] * shapes)
        best_coarse = np.max(lls, axis=-1)
        take_coarse = best_coarse > ll
        return np.where(take_coarse, best_coarse, ll), np.where(take_coarse, coarse[i], amp)

    return _fit(Estimator.GAUSSIAN_TE, wpc.bounds, grid, profile, shape_scale=1.0)
