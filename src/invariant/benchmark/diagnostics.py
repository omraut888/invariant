"""OLS with classical or Newey-West (HAC) standard errors, for validating the benchmark.

Used by the benchmark tests and the size/power Monte Carlo. A small, dependency-free
implementation is preferred over adding statsmodels for a single estimator.

Method: ``beta = (X'X)^{-1} X'y``. Classical covariance ``s^2 (X'X)^{-1}`` with
``s^2 = e'e / (n - k)``. HAC covariance ``(X'X)^{-1} S (X'X)^{-1}`` with the Bartlett-kernel
long-run variance ``S = G_0 + sum_{l=1}^{L} (1 - l / (L + 1)) (G_l + G_l')``,
``G_l = sum_t x_t e_t e_{t-l} x_{t-l}'`` (Newey & West 1987, "A simple, positive semi-definite,
heteroskedasticity and autocorrelation consistent covariance matrix", Econometrica). No
small-sample correction is applied; with ``L = 0`` it reduces to White's HC0. The default lag is
``floor(4 (n / 100)^(2/9))`` (Newey & West 1994).

Assumptions: the classical SE assumes homoskedastic, serially uncorrelated errors; HAC only
requires weak dependence and finite fourth moments, at the cost of finite-sample size
distortion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import numpy.typing as npt

__all__ = ["CovType", "OlsResult", "newey_west_lags", "ols"]

FloatArray = npt.NDArray[np.float64]


class CovType(StrEnum):
    IID = "iid"
    HAC = "hac"


@dataclass(frozen=True)
class OlsResult:
    """Coefficients, SEs and t-stats; index 0 is the intercept when one was added."""

    coef: FloatArray
    se: FloatArray
    tstat: FloatArray
    nobs: int
    cov_type: CovType
    hac_lags: int | None


def newey_west_lags(n: int) -> int:
    """Rule-of-thumb Bartlett bandwidth ``floor(4 (n / 100)^(2/9))``."""
    if n < 1:
        raise ValueError(f"n must be positive, got {n}")
    return math.floor(4.0 * float((n / 100.0) ** (2.0 / 9.0)))


def ols(
    y: FloatArray,
    x: FloatArray,
    *,
    cov: CovType = CovType.IID,
    hac_lags: int | None = None,
    add_constant: bool = True,
) -> OlsResult:
    """Regress ``y`` (n,) on ``x`` (n, k) or (n,). See module docstring for the estimators."""
    y = np.asarray(y, dtype=np.float64)
    design = np.asarray(x, dtype=np.float64)
    if design.ndim == 1:
        design = design[:, None]
    if y.ndim != 1 or design.ndim != 2 or design.shape[0] != y.shape[0]:
        raise ValueError(f"shape mismatch: y {y.shape}, x {design.shape}")
    if add_constant:
        design = np.column_stack([np.ones(len(y)), design])
    n, k = design.shape
    if n <= k:
        raise ValueError(f"need more observations ({n}) than regressors ({k})")
    if not (np.all(np.isfinite(y)) and np.all(np.isfinite(design))):
        raise ValueError("inputs contain non-finite values")
    xtx = design.T @ design
    if np.linalg.matrix_rank(xtx) < k:
        raise ValueError("design matrix is rank deficient")
    xtx_inv = np.linalg.inv(xtx)
    coef = xtx_inv @ (design.T @ y)
    resid = y - design @ coef
    lags: int | None = None
    if cov is CovType.IID:
        vcov = float(resid @ resid) / (n - k) * xtx_inv
    else:
        lags = newey_west_lags(n) if hac_lags is None else hac_lags
        if lags < 0:
            raise ValueError(f"hac_lags must be >= 0, got {lags}")
        xe = design * resid[:, None]
        s = xe.T @ xe
        for lag in range(1, lags + 1):
            g = xe[lag:].T @ xe[:-lag]
            s += (1.0 - lag / (lags + 1.0)) * (g + g.T)
        vcov = xtx_inv @ s @ xtx_inv
    se = np.sqrt(np.diag(vcov))
    with np.errstate(divide="ignore", invalid="ignore"):  # zero SE (exact fit) -> inf / nan t
        tstat = coef / se
    return OlsResult(coef=coef, se=se, tstat=tstat, nobs=n, cov_type=cov, hac_lags=lags)
