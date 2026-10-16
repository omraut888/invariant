"""Conditional-independence (CI) tests for Stage 1, built on tigramite.

Candidates
----------
- ``parcorr``: tigramite ``ParCorr``. Partial correlation of the OLS residuals of X and Y on Z,
  with a Student-t p-value. Assumes linear dependence and i.i.d. Gaussian residuals. Under
  heteroskedasticity the t-test stays valid for a link X -> Y as long as ``X^2`` is
  uncorrelated with ``Var(Y | Z)``, and it is oversized when they are correlated (White 1980).
- ``parcorr_wls``: tigramite ``ParCorrWLS`` with ``expert_knowledge="time-dependent
  heteroskedasticity"``. Each residual is divided by a trailing ``window_size`` mean of the
  absolute OLS residuals before the partial correlation is taken (Günther et al. 2022).
  The weight at t includes ``|resid_t|`` itself, and the first ``window_size - 1`` weights are
  1. Those are properties of tigramite's estimator, kept as is so that we test the published
  method.
- ``parcorr_hc3``: ``ParCorrHC3``. The statistic is ParCorr's partial correlation, but the
  p-value comes from the OLS coefficient of X in ``Y ~ 1 + X + Z`` with White's HC3
  heteroskedasticity-consistent standard error (MacKinnon & White 1985). It needs no variance
  model and no tuning parameter. It is valid under heteroskedasticity of unknown form,
  including Var(Y | X, Z) depending on X (the E4 case). It assumes serially uncorrelated
  errors, which holds for returns (a martingale difference). Asymmetric: Y is the target.
- ``parcorr_wls_centered`` / ``parcorr_wls_centered_loo``: ``CenteredParCorrWLS``, a corrected
  ``ParCorrWLS`` (see "Defect" below). Both regressions include an intercept. The variance is
  estimated from ``|residual|``, not ``|level|``. The trailing window either includes
  ``|resid_t|`` (as tigramite) or excludes it (``leave_current_out``). Rows without a full
  window use the mean of the first window, never tigramite's constant 1. The p-value is
  invariant to affine transforms of X, Y and Z (tests/unit/test_ci_invariance.py).
- ``robust_parcorr``: tigramite ``RobustParCorr``. Each marginal is rank-transformed to normal
  scores before ParCorr (a nonparanormal transform, Liu, Lafferty & Wasserman 2009). This is
  robust to heavy-tailed marginals, not to heteroskedasticity.

- ``parcorr_wls_loo``: ``LeaveCurrentOutParCorrWLS``, our variant of ``parcorr_wls``. The
  weight at t is the mean of ``|resid_{t-w}|, ..., |resid_{t-1}|``, which *excludes*
  ``|resid_t|``. So it is a pure one-step variance forecast (closer to the GARCH ``h_t``) with
  no self-normalisation. The first ``w`` rows, which have no full history, use the mean of the
  first ``w`` residuals. That is a within-sample look-ahead in a test statistic, not in any
  value the pipeline trades on. It exists to test whether ``parcorr_wls``'s calibration comes
  from variance estimation or from including ``|resid_t|`` (ADR 0009, finding 2).

``FastParCorrWLS`` is ``ParCorrWLS`` with one change: the residual computation multiplies by
the weight vector elementwise. tigramite builds a dense ``T x T`` diagonal matrix (O(T^2)
memory and time, ~290 MB at T = 6000). The arithmetic is identical, and the p- and value
matrices of a full PCMCI run match bit for bit (tests/unit/test_ci_tests.py). It is ~45x
faster at T = 6000, and it is what makes the size study (docs/adr/0009) feasible.

Defect (found 2026-10-15, ADR 0009)
-----------------------------------
tigramite 5.2.10.1's ``ParCorrWLS`` is **not location-invariant**:
- ``_get_single_residuals`` defaults to ``standardize=False``, against its own docstring
  ("default: True ... Must be used for partial correlation").
- The regressions have no intercept.
- With an empty Z, the time-dependent std is a window mean of ``|level|``.

A variable with a mean far from 0 is therefore weighted into a near-constant, and the test
degenerates. scripts/repro_tigramite_parcorrwls_location.py reproduces it with tigramite alone.
``FastParCorrWLS`` and ``LeaveCurrentOutParCorrWLS`` inherit the defect. They are kept only to
reproduce studies 1 and 2 and must not be used for decisions.

References: MacKinnon & White (1985), "Some heteroskedasticity-consistent covariance matrix
estimators with improved finite sample properties", Journal of Econometrics 29(3);
Runge et al. (2019), "Detecting and quantifying causal associations in large
nonlinear time series datasets", Science Advances 5(11); Günther, Ninad, Wahl & Runge (2022),
"Conditional independence testing with heteroskedastic data and applications to causal
discovery", NeurIPS 2022; Liu, Lafferty & Wasserman (2009), "The nonparanormal", JMLR 10;
White (1980), "A heteroskedasticity-consistent covariance matrix estimator and a direct test
for heteroskedasticity", Econometrica 48(4).
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from scipy import stats
from tigramite.independence_tests.independence_tests_base import CondIndTest
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.independence_tests.parcorr_wls import ParCorrWLS
from tigramite.independence_tests.robust_parcorr import RobustParCorr

__all__ = [
    "CITestName",
    "CenteredParCorrWLS",
    "FastParCorrWLS",
    "LeaveCurrentOutParCorrWLS",
    "ParCorrHC3",
    "make_ci_test",
]

_FloatArray = npt.NDArray[np.float64]


class CITestName(StrEnum):
    PARCORR = "parcorr"
    PARCORR_WLS = "parcorr_wls"
    PARCORR_WLS_LOO = "parcorr_wls_loo"
    PARCORR_WLS_CENTERED = "parcorr_wls_centered"
    PARCORR_WLS_CENTERED_LOO = "parcorr_wls_centered_loo"
    PARCORR_HC3 = "parcorr_hc3"
    ROBUST_PARCORR = "robust_parcorr"


class FastParCorrWLS(ParCorrWLS):  # type: ignore[misc]  # tigramite is untyped
    """``ParCorrWLS`` with an O(T) residual computation. Inherits the location defect."""

    def _get_single_residuals(
        self,
        array: _FloatArray,
        target_var: int,
        standardize: bool = False,
        return_means: bool = False,
    ) -> _FloatArray | tuple[_FloatArray, _FloatArray | None]:
        # Mirrors ParCorrWLS._get_single_residuals (tigramite 5.2.10.1) line by line, with
        # diag(w) @ M replaced by w[:, None] * M and v @ diag(w) by v * w.
        dim, _ = array.shape
        if np.isnan(np.sum(array)):
            raise ValueError("array has nans")
        stds = cast(_FloatArray, self.stds[target_var])
        if standardize:
            array -= array.mean(axis=1).reshape(dim, 1)
            std = array.std(axis=1)
            for i in range(dim):
                if std[i] != 0.0:
                    array[i] /= std[i]
            if np.isnan(np.sum(array)):
                raise ValueError("array has nans")
        y = np.copy(array[target_var, :])
        weights = np.reciprocal(stds)
        mean: _FloatArray | None
        if dim > 2:
            z = array[2:, :].T.copy()
            beta_hat = np.linalg.lstsq(weights[:, None] * z, y * weights, rcond=None)[0]
            mean = z @ beta_hat
            resid = (y - mean) * weights
            if np.isnan(np.sum(resid)):
                raise ValueError("resid has nans")
        else:
            resid = y * weights
            mean = None
        if return_means:
            return resid, mean
        return resid


class LeaveCurrentOutParCorrWLS(FastParCorrWLS):
    """``FastParCorrWLS`` whose weight at t excludes ``|resid_t|``. Inherits the location defect."""

    def _estimate_std_time(self, arr: _FloatArray, target_var: int) -> _FloatArray:
        # Same OLS residuals as ParCorrWLS._estimate_std_time; only the smoothing differs.
        dim, t_len = arr.shape
        w = int(self.window_size)
        if t_len <= w:
            raise ValueError(f"need more than window_size={w} samples, got {t_len}")
        y = np.copy(arr[target_var, :])
        if dim > 2:
            z = arr[2:, :].T.copy()
            beta_hat = np.linalg.lstsq(z, y, rcond=None)[0]
            resid = np.abs(y - z @ beta_hat)
        else:
            resid = np.abs(y)
        # sums[k] = resid[k] + ... + resid[k + w - 1], each window summed on its own (as in
        # tigramite), so a change at t moves exactly the weights at t+1..t+w.
        sums = np.convolve(resid, np.ones(w), "valid")
        std_est = np.empty(t_len)
        std_est[w:] = sums[:-1] / w  # t >= w: mean of resid[t-w : t]
        std_est[:w] = sums[0] / w
        return std_est


def _with_intercept(z: _FloatArray | None, t_len: int) -> _FloatArray:
    ones = np.ones((t_len, 1))
    return ones if z is None or z.size == 0 else np.hstack([ones, z])


class CenteredParCorrWLS(ParCorrWLS):  # type: ignore[misc]  # tigramite is untyped
    """Location- and scale-invariant ``ParCorrWLS`` (module docstring, "Candidates").

    Only time-dependent heteroskedasticity is supported: the parent-dependent variance model
    of ``ParCorrWLS`` is not reimplemented here.
    """

    # Any: tigramite's CondIndTest takes untyped **kwargs (significance, seed, ...).
    def __init__(self, *, window_size: int, leave_current_out: bool, **kwargs: Any) -> None:
        if window_size < 2:
            raise ValueError(f"window_size must be >= 2, got {window_size}")
        super().__init__(
            expert_knowledge="time-dependent heteroskedasticity", window_size=window_size, **kwargs
        )
        self.leave_current_out = leave_current_out
        self._measure = "par_corr_wls_centered"

    def _estimate_std_time(self, arr: _FloatArray, target_var: int) -> _FloatArray:
        dim, t_len = arr.shape
        w = int(self.window_size)
        if t_len <= w:
            raise ValueError(f"need more than window_size={w} samples, got {t_len}")
        y = arr[target_var, :]
        design = _with_intercept(arr[2:, :].T if dim > 2 else None, t_len)
        beta = np.linalg.lstsq(design, y, rcond=None)[0]
        resid = np.abs(y - design @ beta)
        sums = np.convolve(resid, np.ones(w), "valid")  # sums[k] = resid[k : k + w].sum()
        std_est = np.empty(t_len)
        if self.leave_current_out:
            std_est[w:] = sums[:-1] / w  # mean of resid[t - w : t]
            std_est[:w] = sums[0] / w
        else:
            std_est[w - 1 :] = sums / w  # mean of resid[t - w + 1 : t + 1]
            std_est[: w - 1] = sums[0] / w
        if not np.all(std_est > 0.0):
            raise ValueError("non-positive variance estimate (constant residuals in a window)")
        return std_est

    def _get_single_residuals(
        self,
        array: _FloatArray,
        target_var: int,
        standardize: bool = False,
        return_means: bool = False,
    ) -> _FloatArray | tuple[_FloatArray, _FloatArray]:
        # ``standardize`` is accepted for signature compatibility and ignored: the intercept and
        # the scale-equivariant weights make the residuals affine-invariant without it.
        dim, t_len = array.shape
        if np.isnan(np.sum(array)):
            raise ValueError("array has nans")
        y = array[target_var, :]
        weights = np.reciprocal(cast(_FloatArray, self.stds[target_var]))
        design = _with_intercept(array[2:, :].T if dim > 2 else None, t_len)
        beta = np.linalg.lstsq(weights[:, None] * design, y * weights, rcond=None)[0]
        mean = design @ beta
        resid = (y - mean) * weights
        if return_means:
            return resid, mean
        return resid


class ParCorrHC3(ParCorr):  # type: ignore[misc]  # tigramite is untyped
    """ParCorr's statistic with an HC3 t-test p-value (module docstring, "Candidates")."""

    # Any: tigramite's CondIndTest takes untyped **kwargs (significance, seed, ...).
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._measure = "par_corr_hc3"

    def _get_p_value(
        self,
        val: float,
        array: _FloatArray,
        xyz: npt.NDArray[np.int_],
        T: int,  # noqa: N803  # tigramite's keyword name
        dim: int,
        data_type: object = None,
        sig_override: str | None = None,
    ) -> float:
        if (sig_override or self.significance) != "analytic":
            return float(super()._get_p_value(val, array, xyz, T, dim, data_type, sig_override))
        return hc3_p_value(array)


def hc3_p_value(array: _FloatArray) -> float:
    """Two-sided HC3 p-value of X's coefficient in ``Y ~ 1 + X + Z``.

    ``array`` is tigramite's layout: X in row 0, Y in row 1, Z in rows 2.. (observations in
    columns). Student-t reference with ``T - k`` degrees of freedom (k = 2 + dim Z).
    """
    t_len = array.shape[1]
    design = _with_intercept(array[[0, *range(2, array.shape[0])], :].T, t_len)
    y = array[1, :]
    k = design.shape[1]
    if t_len - k < 1:
        return math.nan
    q, r = np.linalg.qr(design)
    beta = np.linalg.solve(r, q.T @ y)
    resid = y - design @ beta
    leverage = np.sum(q * q, axis=1)
    omega = (resid / (1.0 - leverage)) ** 2
    r_inv = np.linalg.inv(r)
    a = r_inv @ q.T  # (D'D)^{-1} D'
    var_x = float(np.sum(a[1] ** 2 * omega))  # X is column 1 (after the intercept)
    if not var_x > 0.0:
        return math.nan
    t_stat = beta[1] / math.sqrt(var_x)
    return float(2.0 * stats.t.sf(abs(t_stat), t_len - k))


def make_ci_test(name: CITestName, *, seed: int, wls_window: int = 10) -> CondIndTest:
    """A fresh tigramite CI test with analytic significance.

    A new instance is needed per PCMCI run: ``ParCorrWLS`` mutates its ``expert_knowledge`` on
    first use. ``seed`` initialises the test's own ``np.random.default_rng`` (used only by
    shuffle tests and bootstrap CIs, never by analytic p-values). Callers derive it from an
    ``RngFactory`` stream.
    """
    if name is CITestName.PARCORR:
        return ParCorr(significance="analytic", seed=seed)
    # PARCORR_WLS and PARCORR_WLS_LOO carry tigramite's location defect (module docstring):
    # kept to reproduce studies 1-2 only.
    if name is CITestName.PARCORR_HC3:
        return ParCorrHC3(significance="analytic", seed=seed)
    if name in (CITestName.PARCORR_WLS_CENTERED, CITestName.PARCORR_WLS_CENTERED_LOO):
        return CenteredParCorrWLS(
            window_size=wls_window,
            leave_current_out=name is CITestName.PARCORR_WLS_CENTERED_LOO,
            significance="analytic",
            seed=seed,
        )
    if name in (CITestName.PARCORR_WLS, CITestName.PARCORR_WLS_LOO):
        if wls_window < 2:
            raise ValueError(f"wls_window must be >= 2, got {wls_window}")
        cls = FastParCorrWLS if name is CITestName.PARCORR_WLS else LeaveCurrentOutParCorrWLS
        return cls(
            significance="analytic",
            expert_knowledge="time-dependent heteroskedasticity",
            window_size=wls_window,
            seed=seed,
        )
    return RobustParCorr(significance="analytic", seed=seed)
