"""Stylized-facts checker for daily return series (synthetic or real).

Three facts of daily asset returns (Cont 2001, "Empirical properties of asset returns: stylized
facts and statistical issues", Quantitative Finance):

1. **Heavy tails**: positive excess kurtosis (unbiased sample estimator, Fisher definition).
   Caveat: when the 8th moment is infinite (e.g. GARCH-t with nu <= 8, and plausibly real
   returns) the sample kurtosis has no standard CLT and converges slowly, so the threshold is
   deliberately loose and no p-value is reported.
2. **Volatility clustering**: ACF of ``|r|`` positive and significant. Tested with Ljung-Box
   (1978) on ``|r|``, which is valid under the i.i.d. null (|r| then has finite variance), plus a
   minimum mean ACF over lags 1..L so a statistically significant but negligible effect fails.
3. **No linear autocorrelation**: ACF of ``r`` near zero. Under conditional heteroskedasticity
   the Bartlett ``1/sqrt(T)`` band and plain Ljung-Box over-reject, so each lag uses the
   heteroskedasticity-robust standard error ``sqrt(sum x_t^2 x_{t-k}^2) / sum x_t^2`` (Taylor 1984;
   Lo & MacKinlay 1989), with a Bonferroni-corrected two-sided bound over lags 1..L, plus a cap on
   the absolute ACF.

The checker is a pure function returning a frozen report of every number and threshold used.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from scipy import stats

from invariant.core.types import validate_time_index

__all__ = [
    "FactCheck",
    "StylizedFactsReport",
    "StylizedFactsThresholds",
    "acf",
    "check_stylized_facts",
    "robust_acf_se",
]

FloatArray = npt.NDArray[np.float64]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class StylizedFactsThresholds(_Frozen):
    max_lag: int = Field(default=20, ge=1)
    min_excess_kurtosis: float = 1.0
    clustering_alpha: float = Field(default=0.01, gt=0.0, lt=1.0)
    min_mean_abs_acf: float = 0.05
    acf_alpha: float = Field(default=0.01, gt=0.0, lt=1.0)
    max_abs_acf: float = Field(default=0.1, gt=0.0)


class FactCheck(_Frozen):
    name: str
    passed: bool
    statistic: float
    threshold: float
    detail: str


class StylizedFactsReport(_Frozen):
    n_obs: int
    thresholds: StylizedFactsThresholds
    excess_kurtosis: float
    acf: tuple[float, ...]
    acf_robust_t: tuple[float, ...]
    abs_acf: tuple[float, ...]
    abs_ljung_box_q: float
    abs_ljung_box_pvalue: float
    checks: tuple[FactCheck, ...]

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def check(self, name: str) -> FactCheck:
        for c in self.checks:
            if c.name == name:
                return c
        raise KeyError(name)


def acf(x: FloatArray, max_lag: int) -> FloatArray:
    """Sample autocorrelations at lags 1..max_lag (demeaned, denominator ``sum x_t^2``)."""
    d = x - x.mean()
    denom = float(d @ d)
    return np.array([float(d[k:] @ d[:-k]) / denom for k in range(1, max_lag + 1)])


def robust_acf_se(x: FloatArray, max_lag: int) -> FloatArray:
    """Heteroskedasticity-robust SEs of the sample ACF at lags 1..max_lag (Taylor 1984)."""
    d = x - x.mean()
    sq = d * d
    denom = float(sq.sum())
    return np.array(
        [math.sqrt(float(sq[k:] @ sq[:-k])) / denom for k in range(1, max_lag + 1)],
    )


def _ljung_box(r: FloatArray, n: int) -> tuple[float, float]:
    lags = np.arange(1, len(r) + 1)
    q = float(n * (n + 2) * np.sum(r**2 / (n - lags)))
    return q, float(stats.chi2.sf(q, len(r)))


def check_stylized_facts(
    returns: pd.Series[Any],
    thresholds: StylizedFactsThresholds | None = None,
) -> StylizedFactsReport:
    """Check the three stylized facts on a return series (see module docstring).

    ``returns`` must have a valid time index, finite float values, and at least
    ``10 * max_lag`` observations.
    """
    th = thresholds if thresholds is not None else StylizedFactsThresholds()
    validate_time_index(returns)
    x = returns.to_numpy(dtype=np.float64)
    n = len(x)
    lags = th.max_lag
    if n < 10 * lags:
        raise ValueError(f"need at least {10 * lags} observations for max_lag={lags}, got {n}")
    if not np.all(np.isfinite(x)):
        raise ValueError("returns contain non-finite values")
    if float(np.var(x)) == 0.0:
        raise ValueError("returns are constant")

    kurt = float(stats.kurtosis(x, fisher=True, bias=False))
    r = acf(x, lags)
    t = r / robust_acf_se(x, lags)
    abs_r = acf(np.abs(x), lags)
    q, p = _ljung_box(abs_r, n)

    z_crit = float(stats.norm.ppf(1.0 - th.acf_alpha / (2 * lags)))
    max_t = float(np.max(np.abs(t)))
    max_abs_r = float(np.max(np.abs(r)))
    mean_abs_acf = float(abs_r.mean())
    checks = (
        FactCheck(
            name="heavy_tails",
            passed=kurt > th.min_excess_kurtosis,
            statistic=kurt,
            threshold=th.min_excess_kurtosis,
            detail="sample excess kurtosis must exceed threshold",
        ),
        FactCheck(
            name="volatility_clustering",
            passed=p < th.clustering_alpha and mean_abs_acf >= th.min_mean_abs_acf,
            statistic=mean_abs_acf,
            threshold=th.min_mean_abs_acf,
            detail=(
                f"mean ACF(|r|) over lags 1..{lags} must reach threshold and Ljung-Box on |r| "
                f"must reject at {th.clustering_alpha} (Q={q:.1f}, p={p:.3g})"
            ),
        ),
        FactCheck(
            name="no_linear_autocorrelation",
            passed=max_t <= z_crit and max_abs_r <= th.max_abs_acf,
            statistic=max_t,
            threshold=z_crit,
            detail=(
                f"max robust |t| over lags 1..{lags} must not exceed the Bonferroni bound at "
                f"family-wise {th.acf_alpha}; max |ACF| = {max_abs_r:.4f} (cap {th.max_abs_acf})"
            ),
        ),
    )
    return StylizedFactsReport(
        n_obs=n,
        thresholds=th,
        excess_kurtosis=kurt,
        acf=tuple(float(v) for v in r),
        acf_robust_t=tuple(float(v) for v in t),
        abs_acf=tuple(float(v) for v in abs_r),
        abs_ljung_box_q=q,
        abs_ljung_box_pvalue=p,
        checks=checks,
    )
