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
- ``robust_parcorr``: tigramite ``RobustParCorr``. Each marginal is rank-transformed to normal
  scores before ParCorr (a nonparanormal transform, Liu, Lafferty & Wasserman 2009). This is
  robust to heavy-tailed marginals, not to heteroskedasticity.

``FastParCorrWLS`` is ``ParCorrWLS`` with one change: the residual computation multiplies by
the weight vector elementwise. tigramite builds a dense ``T x T`` diagonal matrix (O(T^2)
memory and time, ~290 MB at T = 6000). The arithmetic is identical, and the p- and value
matrices of a full PCMCI run match bit for bit (tests/unit/test_ci_tests.py). It is ~45x
faster at T = 6000, and it is what makes the size study (docs/adr/0009) feasible.

References: Runge et al. (2019), "Detecting and quantifying causal associations in large
nonlinear time series datasets", Science Advances 5(11); Günther, Ninad, Wahl & Runge (2022),
"Conditional independence testing with heteroskedastic data and applications to causal
discovery", NeurIPS 2022; Liu, Lafferty & Wasserman (2009), "The nonparanormal", JMLR 10;
White (1980), "A heteroskedasticity-consistent covariance matrix estimator and a direct test
for heteroskedasticity", Econometrica 48(4).
"""

from __future__ import annotations

from enum import StrEnum
from typing import cast

import numpy as np
import numpy.typing as npt
from tigramite.independence_tests.independence_tests_base import CondIndTest
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.independence_tests.parcorr_wls import ParCorrWLS
from tigramite.independence_tests.robust_parcorr import RobustParCorr

__all__ = ["CITestName", "FastParCorrWLS", "make_ci_test"]

_FloatArray = npt.NDArray[np.float64]


class CITestName(StrEnum):
    PARCORR = "parcorr"
    PARCORR_WLS = "parcorr_wls"
    ROBUST_PARCORR = "robust_parcorr"


class FastParCorrWLS(ParCorrWLS):  # type: ignore[misc]  # tigramite is untyped
    """``ParCorrWLS`` with an O(T) residual computation (see module docstring)."""

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


def make_ci_test(name: CITestName, *, seed: int, wls_window: int = 10) -> CondIndTest:
    """A fresh tigramite CI test with analytic significance.

    A new instance is needed per PCMCI run: ``ParCorrWLS`` mutates its ``expert_knowledge`` on
    first use. ``seed`` initialises the test's own ``np.random.default_rng`` (used only by
    shuffle tests and bootstrap CIs, never by analytic p-values). Callers derive it from an
    ``RngFactory`` stream.
    """
    if name is CITestName.PARCORR:
        return ParCorr(significance="analytic", seed=seed)
    if name is CITestName.PARCORR_WLS:
        if wls_window < 2:
            raise ValueError(f"wls_window must be >= 2, got {wls_window}")
        return FastParCorrWLS(
            significance="analytic",
            expert_knowledge="time-dependent heteroskedasticity",
            window_size=wls_window,
            seed=seed,
        )
    return RobustParCorr(significance="analytic", seed=seed)
